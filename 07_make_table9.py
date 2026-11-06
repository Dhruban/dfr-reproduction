"""Lay out the Waterbirds column of Table 9 (and hence the Waterbirds rows of Table 2) of Kirichenko
et al. next to our numbers, from the pickles written by the authors' dfr_evaluate_spurious.py.

    python 07_make_table9.py --results_dir results/table9 --wb_dir DATA/waterbirds --out results/table9/table9_ours

Expects results/table9/base_seed<N>_dfr.pkl and basenm_seed<N>_dfr.pkl (and optionally
basenm_seed<N>_dfr_noclassweights.pkl).  Worst(%) is the minimum over the four (bird, background)
test groups; Mean(%) weights the group accuracies by their proportions in the training set, as the
paper does (Table 2 caption); the plain test-set mean the authors' script prints is kept in the JSON.
"""
import argparse
import glob
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import pandas as pd

GROUPS = ["landbird_on_land", "landbird_on_water", "waterbird_on_land", "waterbird_on_water"]  # g = 2*y + place
# Table 9, Waterbirds column: (row key, label, paper worst, paper mean)
ROWS = [
    ("base", r"Base (ERM)", "74.9 $\\pm$ 2.4", "98.1 $\\pm$ 0.1"),
    ("dfr_tr", r"$\mathrm{DFR}^{\mathrm{Tr}}_{\mathrm{Tr}}$", "90.2 $\\pm$ 0.8", "97.0 $\\pm$ 0.3"),
    ("dfr_val", r"$\mathrm{DFR}^{\mathrm{Val}}_{\mathrm{Tr}}$", "92.9 $\\pm$ 0.2", "94.2 $\\pm$ 0.4"),
    ("base_nm", r"Base (ERM), trained without minority groups", "31.9 $\\pm$ 3.6", "96.0 $\\pm$ 0.2"),
    ("dfr_tr_nm", r"$\mathrm{DFR}^{\mathrm{Tr}}_{\mathrm{Tr\text{-}NM}}$", "89.9 $\\pm$ 0.6", "94.3 $\\pm$ 0.5"),
    ("dfr_tr_nm_nocw", r"\quad same, no class-weight tuning", "--", "--"),
]


def train_weights(wb_dir):
    meta = pd.read_csv(os.path.join(wb_dir, "metadata.csv"))
    tr = meta[meta["split"] == 0]
    counts = np.array([((tr["y"] == y) & (tr["place"] == p)).sum() for y in (0, 1) for p in (0, 1)], dtype=float)
    return counts / counts.sum()


def metrics(group_accs, w):
    a = np.asarray(group_accs, dtype=float)
    a = a * 100 if a.max() <= 1.0 else a
    return {"group_acc": {g: round(float(v), 2) for g, v in zip(GROUPS, a)},
            "worst": round(float(a.min()), 2), "mean_adjusted": round(float((w * a).sum()), 2)}


def base_groups(res):  # authors' evaluate() dict -> group accuracies in g order
    return [res[f"accuracy_{y}_{p}"] for y in (0, 1) for p in (0, 1)]


def fmt(vals):
    if not vals:
        return "--"
    if len(vals) == 1:
        return f"{vals[0]:.1f}"
    return f"{np.mean(vals):.1f} $\\pm$ {np.std(vals, ddof=1):.1f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/table9")
    p.add_argument("--wb_dir", required=True)
    p.add_argument("--out", default="results/table9/table9_ours")
    args = p.parse_args()
    w = train_weights(args.wb_dir)

    per_seed = defaultdict(dict)  # row key -> seed -> metrics
    for f in sorted(glob.glob(os.path.join(args.results_dir, "*_dfr*.pkl"))):
        m = re.match(r"(base|basenm)_seed(\d+)_dfr(_noclassweights)?\.pkl$", os.path.basename(f))
        if not m:
            continue
        kind, seed, nocw = m.group(1), int(m.group(2)), bool(m.group(3))
        with open(f, "rb") as fh:
            r = pickle.load(fh)
        base = metrics(base_groups(r["base_model_results"]["test"]), w)
        base["mean_plain"] = round(100 * float(r["base_model_results"]["test"]["mean_accuracy"]), 2)
        dfr_tr = metrics(r["dfr_train_results"]["test_accs"], w)
        dfr_tr["mean_plain"] = round(100 * float(r["dfr_train_results"]["test_mean_acc"]), 2)
        dfr_tr["hypers_C_w0_w1"] = [float(x) for x in r["dfr_train_results"]["best_hypers"]]
        dfr_val = metrics(r["dfr_val_results"]["test_accs"], w)
        dfr_val["mean_plain"] = round(100 * float(r["dfr_val_results"]["test_mean_acc"]), 2)
        dfr_val["hypers_C_w0_w1"] = [float(x) for x in r["dfr_val_results"]["best_hypers"]]
        if kind == "base":
            per_seed["base"][seed], per_seed["dfr_tr"][seed], per_seed["dfr_val"][seed] = base, dfr_tr, dfr_val
        elif nocw:
            per_seed["dfr_tr_nm_nocw"][seed] = dfr_tr
        else:
            per_seed["base_nm"][seed], per_seed["dfr_tr_nm"][seed] = base, dfr_tr
            per_seed["dfr_val_nm"][seed] = dfr_val  # not in the paper's table; kept in the JSON

    with open(args.out + ".json", "w") as fh:
        json.dump(per_seed, fh, indent=2)

    lines = [r"\begin{table}[h]", r"\centering", r"\begin{tabular}{l cc cc}", r"\hline",
             r" & \multicolumn{2}{c}{Original paper (Table 9)} & \multicolumn{2}{c}{Ours} \\",
             r"Method & Worst (\%) & Mean (\%) & Worst (\%) & Mean (\%) \\", r"\hline"]
    csv = ["method,paper_worst,paper_mean,ours_worst,ours_mean,n_seeds"]
    for key, label, pw, pm in ROWS:
        if key == "base_nm":
            lines.append(r"\hline")
        seeds = per_seed.get(key, {})
        worst = [seeds[s]["worst"] for s in sorted(seeds)]
        mean = [seeds[s]["mean_adjusted"] for s in sorted(seeds)]
        lines.append(f"{label} & {pw} & {pm} & {fmt(worst)} & {fmt(mean)} \\\\")
        csv.append(f"{key},{pw.replace('$', '').replace(chr(92) + 'pm', '+-')},{pm.replace('$', '').replace(chr(92) + 'pm', '+-')},"
                   f"{np.mean(worst) if worst else ''},{np.mean(mean) if mean else ''},{len(worst)}")
    n = sorted({len(v) for v in per_seed.values() if v})
    seed_note = "" if not n else (f" ({n[0]} seed)" if n == [1] else f" ({n[0]}--{n[-1]} seeds, mean $\\pm$ std)")
    lines += [r"\hline", r"\end{tabular}",
              r"\caption{Waterbirds worst-group and mean test accuracy of the ERM base model and the DFR "
              r"variants. \textbf{Left: Table~9 of the original paper} (mean $\pm$ std over 5 runs). "
              r"\textbf{Right: ours}" + seed_note + r", produced with the authors' \texttt{dfr\_evaluate\_spurious.py}. "
              r"Mean accuracy weights the four groups by their training-set proportions, as in the paper.}",
              r"\label{tab:table9}", r"\end{table}"]
    open(args.out + ".tex", "w").write("\n".join(lines) + "\n")
    open(args.out + ".csv", "w").write("\n".join(csv) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}.tex, .csv and .json")


if __name__ == "__main__":
    main()
