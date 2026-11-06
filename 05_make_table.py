"""Collect the JSON files written by 04_eval_table1.py into Table 1 (paper numbers side by side
with ours) as LaTeX and CSV.

    python 05_make_table.py --results_dir results/table1 --out table1_ours

Runs are matched to rows by the prefix of their name (original95_seed1 -> original95, ...).
With several seeds per row the table shows mean +- std; with one seed just the number.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

# Table 1 of Kirichenko et al. (ICLR 2023): worst-group test accuracy (%)
ROWS = [  # (run-name prefix, label in the table, paper Original, paper FG-Only)
    ("balanced", r"Balanced (50\%), 256-px backgrounds", 91.9, 94.7),
    ("balanced_hires", r"Balanced (50\%), full-res backgrounds", 91.9, 94.7),
    ("original95", r"Original (95\%)", 73.8, 93.7),
    ("original100", r"Original (100\%)", 38.4, 94.0),
    ("fgonly", "FG-Only (--)", 75.2, 95.5),
]


def fmt(vals):
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{np.mean(vals):.1f} $\\pm$ {np.std(vals, ddof=1):.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/table1")
    p.add_argument("--out", default="table1_ours", help="basename for the .tex and .csv outputs")
    p.add_argument("--metric", default="worst_group_acc",
                   choices=["worst_group_acc", "worst_class_acc", "mean_acc", "adjusted_mean_acc"])
    args = p.parse_args()

    ours = defaultdict(lambda: {"original_test": [], "fg_only_test": [], "names": []})
    for f in sorted(glob.glob(os.path.join(args.results_dir, "*.json"))):
        r = json.load(open(f))
        prefix = r["name"].split("_seed")[0]
        for s in ("original_test", "fg_only_test"):
            ours[prefix][s].append(r["sets"][s][args.metric])
        ours[prefix]["names"].append(r["name"])

    lines = [
        r"\begin{table}[h]", r"\centering",
        r"\begin{tabular}{l cc cc}", r"\hline",
        r" & \multicolumn{2}{c}{Original paper (Table 1)} & \multicolumn{2}{c}{Ours} \\",
        r"Train data (spur.\ corr.) & Original & FG-Only & Original & FG-Only \\", r"\hline",
    ]
    csv = ["train_data,paper_original,paper_fg_only,ours_original,ours_fg_only,n_seeds,runs"]
    for prefix, label, p_o, p_fg in ROWS:
        o, fg = ours[prefix]["original_test"], ours[prefix]["fg_only_test"]
        lines.append(f"{label} & {p_o:.1f} & {p_fg:.1f} & {fmt(o)} & {fmt(fg)} \\\\")
        csv.append(f"{prefix},{p_o},{p_fg},{np.mean(o) if o else ''},{np.mean(fg) if fg else ''},"
                   f"{len(o)},{' '.join(ours[prefix]['names'])}")
    n_seeds = sorted({len(ours[r[0]]["original_test"]) for r in ROWS if ours[r[0]]["original_test"]})
    if not n_seeds:
        seed_note = "."
    elif n_seeds[0] == n_seeds[-1]:
        seed_note = f" ({n_seeds[0]} seed" + (", mean $\\pm$ std)." if n_seeds[0] > 1 else ").")
    else:
        seed_note = f" ({n_seeds[0]}--{n_seeds[-1]} seeds per row; mean $\\pm$ std where more than one)."
    lines += [
        r"\hline", r"\end{tabular}",
        r"\caption{Worst-group test accuracy (\%) on Waterbirds for ERM ResNet-50 models trained on "
        r"variants of the training data and evaluated on the Original and FG-Only test sets. The two Balanced "
        r"rows differ only in the resolution of the Places photos pasted behind the 2,157 re-backgrounded "
        r"training images (same images flipped, same seed); the paper's number is repeated for both. "
        r"\textbf{Left: numbers reported in Table~1 of the original paper} (single run). "
        r"\textbf{Right: our reproduction}" + seed_note + r" Rows marked -- were not run.}",
        r"\label{tab:table1}", r"\end{table}",
    ]
    tex = "\n".join(lines)
    open(args.out + ".tex", "w").write(tex + "\n")
    open(args.out + ".csv", "w").write("\n".join(csv) + "\n")
    print(tex)
    print(f"\nwrote {args.out}.tex and {args.out}.csv  (metric: {args.metric})")


if __name__ == "__main__":
    main()
