#!/bin/bash
# Result 2: the Waterbirds column of Table 9 of Kirichenko et al. (which contains the two Waterbirds
# rows of Table 2).  Everything is the authors' code:
#   * train_classifier.py with the Appendix C recipe (100 epochs, weight decay 1e-3) -> base ERM model
#     - task 0: standard training set                      -> Base (ERM), DFR_Tr^Tr, DFR_Tr^Val
#     - task 1: training set without the 2 minority groups -> Base (ERM) NM, DFR_Tr^Tr-NM
#   * dfr_evaluate_spurious.py on each checkpoint          -> all DFR variants (results pickle)
# plus our 04_eval_table1.py to cache the penultimate-layer embeddings (for the experiments beyond the
# paper) and 07_make_table9.py to lay the numbers out next to the paper's.
#
#   export DATA_ROOT=... ; export REPO=... ; export SEED=1
#   sbatch --account=<ACCOUNT> 06_train_dfr_table9.sbatch          # both tasks
#   sbatch --account=<ACCOUNT> --array=0 06_train_dfr_table9.sbatch # only the standard base model
#
# Outputs: logs/table9/<base|basenm>_seed<SEED>/final_checkpoint.pt, results/table9/<run>_dfr.pkl,
# results/table9/<run>.json (+ _embeddings.npz), results/table9/table9_ours.{tex,csv}.
#SBATCH --job-name=dfr_table9
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=5
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --array=0-1
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_logs/table9_%A_%a.out
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT}"; : "${REPO:?set REPO}"; SEED=${SEED:-1}

module load python3.10-anaconda/2023.03
source ~/dfr_env/bin/activate

NAMES=(base basenm)
EXTRA=("" "--num_minority_groups_remove=2")

i=${SLURM_ARRAY_TASK_ID}
NAME=${NAMES[$i]}
RUN=${NAME}_seed${SEED}
OUT=$SLURM_SUBMIT_DIR/logs/table9/$RUN
RESULTS=$SLURM_SUBMIT_DIR/results/table9
mkdir -p "$OUT" "$RESULTS"

echo "[$(date)] start ${RUN} on $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
START=$(date +%s)

# ---- 1. base model, Appendix C recipe (the README's Waterbirds command) ------------------------------
cd "$REPO"
python train_classifier.py \
  --output_dir="$OUT" \
  --data_dir="$DATA_ROOT/waterbirds" \
  --test_wb_dir="$DATA_ROOT/waterbirds" \
  --test_grey_dir="$DATA_ROOT/waterbirds_fg" \
  --pretrained_model --augment_data \
  --num_epochs=100 --weight_decay=1e-3 --batch_size=32 --init_lr=1e-3 \
  --eval_freq=100 --seed="$SEED" ${EXTRA[$i]}
T1=$(date +%s)
echo "[$(date)] trained ${RUN}: $(( (T1-START)/60 )) min wall-clock" | tee "$OUT/walltime.txt"

# ---- 2. DFR (authors' script): base-model evaluation, embedding extraction, DFR_Tr^Val and DFR_Tr^Tr ---
# --tune_class_weights_dfr_train is the Appendix C setting for DFR_Tr^Tr.  For the no-minority-group
# model the README says the flag may be omitted; we run that model both ways and keep both pickles.
python dfr_evaluate_spurious.py --data_dir="$DATA_ROOT/waterbirds" \
  --ckpt_path="$OUT/final_checkpoint.pt" --result_path="$RESULTS/${RUN}_dfr.pkl" \
  --tune_class_weights_dfr_train 2>&1 | tail -25
if [ "$NAME" = basenm ]; then
  python dfr_evaluate_spurious.py --data_dir="$DATA_ROOT/waterbirds" \
    --ckpt_path="$OUT/final_checkpoint.pt" --result_path="$RESULTS/${RUN}_dfr_noclassweights.pkl" 2>&1 | tail -25
fi
T2=$(date +%s)
echo "[$(date)] DFR for ${RUN}: $(( (T2-T1)/60 )) min wall-clock" | tee -a "$OUT/walltime.txt"

# ---- 3. our evaluation of the base model on both test sets + embedding cache ---------------------------
cd "$SLURM_SUBMIT_DIR"
python 04_eval_table1.py --repo_dir "$REPO" \
  --ckpt "$OUT/final_checkpoint.pt" --name "$RUN" \
  --wb_dir "$DATA_ROOT/waterbirds" --fg_dir "$DATA_ROOT/waterbirds_fg" \
  --out_dir "$RESULTS" --save_embeddings

# ---- 4. table next to the paper's numbers -----------------------------------------------------------------
python 07_make_table9.py --results_dir "$RESULTS" --wb_dir "$DATA_ROOT/waterbirds" --out "$RESULTS/table9_ours"
echo "[$(date)] done ${RUN}"
