#!/bin/bash
# Train the ERM models of Table 1 (Kirichenko et al., Appendix B.1 recipe) as a SLURM job array
# on Great Lakes with the authors' unmodified train_classifier.py, and evaluate each model on the
# Original and FG-Only test sets as soon as it finishes, so results arrive one row at a time.
#
#   export DATA_ROOT=/scratch/<account>_root/<account>/<uniqname>/dfr_data
#   export REPO=$HOME/deep_feature_reweighting      # authors' repo
#   export SEED=1
#   mkdir -p slurm_logs                              # SLURM will not create it
#   sbatch --account=<ACCOUNT> 03_train_table1.sbatch                 # all four rows
#   sbatch --account=<ACCOUNT> --array=0,1,3 03_train_table1.sbatch   # the three rows that need no Places
#   sbatch --account=<ACCOUNT> --array=2 03_train_table1.sbatch       # Balanced, once 02_make_balanced.py ran
#
# Array index -> Table 1 row:
#   0  original95      Original (95%)   : standard training set
#   1  original100     Original (100%)  : standard training set minus the two minority groups
#   2  balanced        Balanced (50%)   : output of 02_make_balanced.py with the 256-px Places release
#   3  fgonly          FG-Only          : output of 01_make_fg_only.py
#   4  balanced_hires  Balanced (50%)   : output of 02_make_balanced.py with the full-resolution Places
#                                         release (what Sagawa et al. used); run with --array=4
#
# After each model trains, the same job runs 04_eval_table1.py on its checkpoint (writes
# results/table1/<row>_seed<SEED>.json and the cached embeddings) and 05_make_table.py (rewrites
# results/table1/table1_ours.tex/.csv with every row finished so far).  Wall-clock per run is in
# logs/table1/<row>_seed<SEED>/walltime.txt and in `sacct -j <jobid> --format=JobID,JobName,Elapsed,State`.
#SBATCH --job-name=dfr_table1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/table1_%A_%a.out
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT}"; : "${REPO:?set REPO}"; SEED=${SEED:-1}

# --- environment (Great Lakes): python module + venv with torch==2.14.0+cu126 torchvision==0.29.0+cu126
#     scikit-learn pandas tensorboard tqdm pillow.  The CUDA-12.6 build is required for V100 nodes.
module load python3.10-anaconda/2023.03
source ~/dfr_env/bin/activate

NAMES=(original95 original100 balanced fgonly balanced_hires)
TRAIN_DIRS=("$DATA_ROOT/waterbirds" "$DATA_ROOT/waterbirds" "$DATA_ROOT/waterbirds_balanced" "$DATA_ROOT/waterbirds_fg" "$DATA_ROOT/waterbirds_balanced_hires")
EXTRA=("" "--num_minority_groups_remove=2" "" "" "")

i=${SLURM_ARRAY_TASK_ID}
NAME=${NAMES[$i]}
RUN=${NAME}_seed${SEED}
OUT=$SLURM_SUBMIT_DIR/logs/table1/$RUN
RESULTS=$SLURM_SUBMIT_DIR/results/table1
mkdir -p "$OUT" "$RESULTS"

echo "[$(date)] start ${RUN} on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
START=$(date +%s)

# ---- 1. train (authors' code, Appendix B.1 hyper-parameters) ----------------------------------
# ImageNet-pretrained ResNet-50, 50 epochs, SGD lr 1e-3, momentum 0.9, batch 32, weight decay 1e-2,
# RandomResizedCrop(224, scale=(0.7,1)) + horizontal flip (= --augment_data), no early stopping.
# --eval_freq=50 skips the in-training test evaluation (only epoch 0 is evaluated); the final
# checkpoint is scored below instead.
cd "$REPO"
python train_classifier.py \
  --output_dir="$OUT" \
  --data_dir="${TRAIN_DIRS[$i]}" \
  --test_wb_dir="$DATA_ROOT/waterbirds" \
  --test_grey_dir="$DATA_ROOT/waterbirds_fg" \
  --pretrained_model --augment_data \
  --num_epochs=50 --weight_decay=1e-2 --batch_size=32 --init_lr=1e-3 \
  --eval_freq=50 --seed="$SEED" ${EXTRA[$i]}

END=$(date +%s)
echo "[$(date)] trained ${RUN}: $(( (END-START)/60 )) min wall-clock" | tee "$OUT/walltime.txt"

# ---- 2. evaluate this model on the Original and FG-Only test sets -----------------------------
cd "$SLURM_SUBMIT_DIR"
python 04_eval_table1.py --repo_dir "$REPO" \
  --ckpt "$OUT/final_checkpoint.pt" --name "$RUN" \
  --wb_dir "$DATA_ROOT/waterbirds" --fg_dir "$DATA_ROOT/waterbirds_fg" \
  --out_dir "$RESULTS" --save_embeddings

# ---- 3. refresh the table with every row finished so far -------------------------------------
python 05_make_table.py --results_dir "$RESULTS" --out "$RESULTS/table1_ours"
echo "[$(date)] done ${RUN}; current table: $RESULTS/table1_ours.csv"
cat "$RESULTS/table1_ours.csv"
