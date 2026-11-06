# Reproducing Table 1 of Kirichenko, Izmailov & Wilson (ICLR 2023)

Table 1: worst-group test accuracy of ERM ResNet-50 models trained on four variants of
Waterbirds (Balanced 50%, Original 95%, Original 100%, FG-Only) and evaluated on the Original
and the FG-Only test set.

Everything that trains a network is the authors' own, unmodified `train_classifier.py`
(github.com/PolinaKirichenko/deep_feature_reweighting). The scripts here supply what the repo
does not have: the FG-Only and Balanced data aligned to the standard Waterbirds tarball, and an
evaluator that scores a checkpoint on two test sets.

## Files

| file | what it does | needs |
|---|---|---|
| `00_download_data.sh` | Waterbirds tarball, CUB-200-2011 + segmentation masks, the 4 Places365 background categories | network |
| `wb_compose.py` | compositing helpers (ported from group_DRO `dataset_utils.py`) | – |
| `01_make_fg_only.py` | FG-Only images (bird on grey 150) for train/val/test, same `metadata.csv` as the tarball; alignment checks | CUB (+ Places for the optional pixel check) |
| `02_make_balanced.py` | Balanced (50%) training set: flips the background of the minimal random subset of training images (App. B.1) | CUB + Places |
| `03_train_table1.sbatch` | SLURM job array, one task per Table 1 row, Appendix B.1 recipe; each task evaluates its own model when done and refreshes the table | GPU |
| `04_eval_table1.py` | per-group / worst-group / mean accuracy of a checkpoint on Original and FG-Only test; optional embedding cache (called by the sbatch; can also be run by hand) | GPU or CPU |
| `05_make_table.py` | LaTeX + CSV table, paper numbers next to ours; rows not yet run show as `--` | – |

## Run order

```bash
# 0. environment (once).  Any recent torch/torchvision works; the repo has no requirements.txt.
python -m venv ~/dfr_env && source ~/dfr_env/bin/activate
pip install torch torchvision scikit-learn pandas tensorboard tqdm pillow
git clone https://github.com/PolinaKirichenko/deep_feature_reweighting ~/deep_feature_reweighting

export DATA_ROOT=/scratch/<account_root>/<uniqname>/dfr_data   # use scratch, ~35 GB peak with Places-small
export REPO=~/deep_feature_reweighting

# 1. data.  Places (24 GB) is only needed for the Balanced row; `none` skips it, `small` = 256 px,
#    `large` = full resolution (what Sagawa et al. used, 105 GB).  The script skips what exists,
#    so you can run it with `none` first and with `small` later.
bash 00_download_data.sh $DATA_ROOT none

# 2. FG-Only data (needs only CUB).  Add --places_dir .../places/data_256 --n_check 50 once Places
#    exists to also run the pixel-level check that our compositing reproduces the tarball.
python 01_make_fg_only.py --wb_dir $DATA_ROOT/waterbirds --cub_dir $DATA_ROOT/CUB_200_2011 \
    --out_dir $DATA_ROOT/waterbirds_fg --workers 8

# 3. train + evaluate.  Each array task trains one model, scores it on both test sets as soon as it
#    is done, and rewrites results/table1/table1_ours.{tex,csv}, so rows appear one at a time.
#    (Edit the module/venv lines in the sbatch first.)
mkdir -p slurm_logs
export SEED=1
sbatch --account=<ACCOUNT> --array=0,1,3 03_train_table1.sbatch   # Original 95%, Original 100%, FG-Only

# 4. meanwhile: Places, Balanced data, then the Balanced row
bash 00_download_data.sh $DATA_ROOT small
python 02_make_balanced.py --wb_dir $DATA_ROOT/waterbirds --cub_dir $DATA_ROOT/CUB_200_2011 \
    --places_dir $DATA_ROOT/places/data_256 --out_dir $DATA_ROOT/waterbirds_balanced --seed 0
sbatch --account=<ACCOUNT> --array=2 03_train_table1.sbatch

# at any time: the current table
cat results/table1/table1_ours.csv
```

On the cluster, run steps 1, 2 and 4 as batch jobs (`sbatch --wrap="..."`) chained with
`--dependency=afterok:<jobid>`, not on the login node.  More seeds: `SEED=2 sbatch ...` etc.;
`05_make_table.py` averages everything with the same row prefix.  `04_eval_table1.py --runs_dir
logs/table1` re-evaluates every finished run by hand if ever needed.

## What to expect (time)

| step | wall-clock | notes |
|---|---|---|
| Waterbirds tarball + CUB | 10–20 min | ~4 GB |
| Places-small download + extracting 4 categories | 30–90 min | 24 GB download, one pass over the tar; unattended |
| `01_make_fg_only.py` (11,788 images, 8 workers) | 1–3 min | |
| `02_make_balanced.py` (~2,150 composites + links) | 1–2 min | |
| one training run (50 epochs, batch 32, 4,795 images) | 30–60 min on a V100/A100 | plus queue wait; array tasks run in parallel only if the account allows more than one GPU at a time |
| evaluation at the end of each training task | ~2–3 min on GPU | both test sets + embedding cache for train/val/test |

Record the per-run wall-clock from `logs/table1/<run>/walltime.txt` or
`sacct -j <jobid> --format=JobID,JobName,Elapsed,State` for the report.

