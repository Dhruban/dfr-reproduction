# Reproducing *Last Layer Re-Training is Sufficient for Robustness to Spurious Correlations*

Kirichenko, Izmailov & Wilson, ICLR 2023 ([arXiv:2204.02937](https://arxiv.org/abs/2204.02937); authors' code: [PolinaKirichenko/deep_feature_reweighting](https://github.com/PolinaKirichenko/deep_feature_reweighting), commit in `environment/authors_repo_commit.txt`).

EECS 553 (Fall 2026, University of Michigan) reproducibility project. Team: *(names)*.

## What is reproduced

| | paper | what we run | status |
|---|---|---|---|
| Result 1 | **Table 1** — Waterbirds ERM models trained on Balanced (50%), Original (95%), Original (100%) and FG-Only data, each evaluated on the Original and the FG-Only test set | all four rows, plus an ablation on the resolution of the Places backgrounds used to build the Balanced set | seed 1 done |
| Result 2 | **Table 2 / Table 9, Waterbirds column** — Base (ERM), DFR_Tr^Tr, DFR_Tr^Val, Base (ERM) trained without minority groups, DFR_Tr-NM^Tr | all five rows with the authors' `dfr_evaluate_spurious.py`, plus a class-weight ablation on the NM row | seed 1 done |
| Beyond the paper | — | DFR_Tr^Val as a function of the size of the group-labelled validation set (5–133 minority images per group), with the paper's l1 penalty and with l2 | done (base model seed 1) |

All networks are trained with the authors' unmodified `train_classifier.py`; all DFR numbers come from their unmodified `dfr_evaluate_spurious.py`. Our scripts provide what the authors' repository does not: data preparation aligned with the public Waterbirds release, cluster job scripts, a two-test-set evaluator, table builders and the size sweep.

## Repository layout

```
00_download_data.sh         Waterbirds tarball (if reachable), CUB-200-2011 + masks, Places365 (4 categories)
00b_waterbirds_from_hf.py   rebuild the standard Waterbirds folder from the Hugging Face mirror; verifies group counts
wb_compose.py               compositing helpers ported from group_DRO/dataset_scripts/dataset_utils.py
01_make_fg_only.py          FG-Only images (bird on grey 150), aligned 1:1 with the public release
02_make_balanced.py         Balanced (50%) training set: minimal random background flips (paper, App. B.1)
03_train_table1.sbatch      Table 1 models (App. B.1 recipe), one Slurm array task per row; evaluates each when done
04_eval_table1.py           per-group / worst-group / mean accuracy of a checkpoint on both test sets; embedding cache
05_make_table.py            Table 1 side by side with the paper (LaTeX + CSV)
06_train_dfr_table9.sbatch  Table 9 base models (App. C recipe) + authors' DFR script + evaluation + table
07_make_table9.py           Table 2 / 9 Waterbirds column side by side with the paper
08_size_sweep.py            result beyond the paper (CPU only, uses the cached embeddings)
PIPELINE.md                 file-by-file notes, run order, timings, things to state in the report
results/table1, table9, beyond   result JSONs, DFR pickles, tables (.csv/.tex/.json), sweep CSVs and figure
logs/table1, table9         one folder per training run: log.txt, args.json, command.sh, walltime.txt
slurm_logs/                 Slurm job logs
environment/                pip freeze, library versions, authors' repo commit
```

Checkpoints (`*.pt`) and embedding caches (`*.npz`) are not in git (size); see *Large artefacts*.

## How to reproduce

Tested on the U-M Great Lakes cluster (V100 nodes, account limit of one GPU at a time). Python 3.10, `torch 2.14.0+cu126`, `torchvision 0.29.0` — the CUDA 12.6 build is required on V100s; exact versions in `environment/requirements_lock.txt`.

```bash
# environment
python -m venv ~/dfr_env && source ~/dfr_env/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install scikit-learn pandas tensorboard tqdm pillow huggingface_hub pyarrow matplotlib
git clone https://github.com/PolinaKirichenko/deep_feature_reweighting ~/deep_feature_reweighting
export REPO=~/deep_feature_reweighting DATA_ROOT=/scratch/<...>/dfr_data ACCOUNT=<slurm account>

# data (login node; compute nodes have no outbound access on Great Lakes)
bash 00_download_data.sh $DATA_ROOT large          # CUB + full-resolution Places; Waterbirds tarball if nlp.stanford.edu is up
python -c "from huggingface_hub import snapshot_download; snapshot_download('grodino/waterbirds', repo_type='dataset', local_dir='$DATA_ROOT/hf_waterbirds')"
python 00b_waterbirds_from_hf.py --hf_dir $DATA_ROOT/hf_waterbirds --out_dir $DATA_ROOT/waterbirds --cub_dir $DATA_ROOT/CUB_200_2011
python 01_make_fg_only.py --wb_dir $DATA_ROOT/waterbirds --cub_dir $DATA_ROOT/CUB_200_2011 --out_dir $DATA_ROOT/waterbirds_fg --workers 8
python 02_make_balanced.py --wb_dir $DATA_ROOT/waterbirds --cub_dir $DATA_ROOT/CUB_200_2011 --places_dir $DATA_ROOT/places/data_large --out_dir $DATA_ROOT/waterbirds_balanced_hires --seed 0

# result 1 (Table 1): rows 0,1,3,4 = Original 95%, Original 100%, FG-Only, Balanced (full-res); row 2 = Balanced with 256-px Places (ablation)
mkdir -p slurm_logs; SEED=1 sbatch --account=$ACCOUNT --array=0,1,3,4 03_train_table1.sbatch
# result 2 (Table 2 / 9): task 0 = standard base model, task 1 = no-minority-groups base model
SEED=1 sbatch --account=$ACCOUNT 06_train_dfr_table9.sbatch
# beyond the paper (CPU, ~1 min)
python 08_size_sweep.py --npz results/table9/base_seed1_embeddings.npz --out_dir results/beyond --jobs 8
```

More seeds: `SEED=2 sbatch ...`; the table scripts average all seeds present. Run every `sbatch` from the repository root (the job scripts locate `logs/` and `results/` through `$SLURM_SUBMIT_DIR`).

## Results so far

**Table 1** (worst-group test accuracy %, Original test / FG-Only test; paper = single run, ours = seed 1):

| training data | paper | ours |
|---|---|---|
| Balanced (50%) — full-resolution Places backgrounds | 91.9 / 94.7 | 90.2 / 89.9 |
| Balanced (50%) — 256-px Places backgrounds (ablation) | — | 83.0 / 83.8 |
| Original (95%) | 73.8 / 93.7 | 60.0 / 91.7 |
| Original (100%) | 38.4 / 94.0 | 35.7 / 90.0 |
| FG-Only | 75.2 / 95.5 | 73.5 / 91.1 |

**Table 2 / Table 9, Waterbirds** (worst-group / mean %; paper = mean ± std over 5 runs, ours = seed 1):

| method | paper | ours |
|---|---|---|
| Base (ERM) | 74.9 ± 2.4 / 98.1 ± 0.1 | 76.8 / 98.2 |
| DFR_Tr^Tr | 90.2 ± 0.8 / 97.0 ± 0.3 | 91.8 / 96.3 |
| DFR_Tr^Val | 92.9 ± 0.2 / 94.2 ± 0.4 | 92.3 / 95.6 |
| Base (ERM), trained without minority groups | 31.9 ± 3.6 / 96.0 ± 0.2 | 26.4 / 95.9 |
| DFR_Tr-NM^Tr | 89.9 ± 0.6 / 94.3 ± 0.5 | 90.6 / 93.4 |
| DFR_Tr-NM^Tr, class weights not tuned (ablation) | — | 87.4 / 92.2 |

**Size sweep** (DFR_Tr^Val on validation sets shrunk to *n* minority images per group, composition preserved; worst-group % mean ± std over 10 random shrinkages; base model seed 1; ERM base model 76.8):

| n (images available) | 5 (46) | 10 (90) | 20 (180) | 35 (316) | 50 (451) | 75 (676) | 100 (901) | 133 (1199) |
|---|---|---|---|---|---|---|---|---|
| l1 (paper's penalty) | 85.7 ± 3.2 | 88.3 ± 1.9 | 89.3 ± 2.1 | 90.7 ± 1.1 | 91.3 ± 1.2 | 91.4 ± 1.1 | 91.9 ± 0.5 | 92.5 ± 0.4 |
| l2 | 87.2 ± 3.6 | 88.7 ± 2.4 | 91.1 ± 1.2 | 91.2 ± 1.3 | 92.4 ± 1.0 | 92.5 ± 0.9 | 92.9 ± 0.6 | 93.5 ± 0.2 |

Full tables including mean accuracy: `results/*/`*.csv / *.tex; figure: `results/beyond/size_sweep.pdf`.

## Compute and time

One V100: Table 1 models ≈ 16 min each (50 epochs); Table 9 base models ≈ 28 min each (100 epochs) + ≈ 5 min for the DFR script; size sweep ≈ 40 s on 8 CPUs. Per-run wall-clock in `logs/*/*/walltime.txt`; Slurm records in `slurm_logs/`.

## Fidelity notes

- **Data.** The public Waterbirds release (Sagawa et al., 2020), obtained from the Hugging Face mirror `grodino/waterbirds` because `nlp.stanford.edu` was unreachable; all 12 split/label/background group counts verified against the official release. For Table 1 the authors regenerated Waterbirds themselves (App. B.1), which the group_DRO README states does not reproduce the released dataset; their regenerated set is not public, so FG-Only and Balanced are derived here from the public release following App. B.1.
- **Balanced set.** Built as described in App. B.1 (1,657 landbirds land→water, 500 waterbirds water→land). With 256-px Places backgrounds the resulting model scores 83.0 / 83.8; with the full-resolution release (what the original dataset was built from) 90.2 / 89.9 — same images flipped, same seed.
- **Code vs paper text.** The authors' DFR script averages 20 refits (paper: 10) and standardises the final DFR_Tr^Val fit with training-set statistics (paper: reweighting data). Mean accuracy is computed by weighting group accuracies by their training-set proportions, as the paper states; the script prints the plain average, which we do not report.
- **Class weights for DFR_Tr-NM^Tr.** The paper describes class-weight tuning for DFR_Tr^Tr only; the README leaves the flag optional for the NM model. We run both: with tuning the result matches the paper's number; without it is ≈ 3 points lower.
- **Software.** torch 2.14 / CUDA 12.6 / V100, 2026; the authors do not pin versions (repository dated April 2022, notebooks run with Python 3.8.5).

## Large artefacts (not in git)

Checkpoints (`logs/*/*/final_checkpoint.pt`, ≈ 95 MB each) and embedding caches (`results/*/*_embeddings.npz`) regenerate every table exactly; they are archived at *(link)*. Retraining from scratch with the same seed reproduces the numbers up to GPU run-to-run variation.

## Use of generative AI

Claude (Anthropic) was used to plan the reproduction, write the scripts in this repository and draft report text, with transcripts kept for the course's GenAI disclosure report. All experiments were run and all numbers checked by the team.
