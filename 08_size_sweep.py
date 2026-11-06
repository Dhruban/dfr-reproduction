"""Result beyond the paper: how many group-labelled reweighting examples does DFR need?

The paper always uses the whole validation set (467/466/133/133 images per group on Waterbirds,
group-balanced to 133 per group) as the reweighting data for DFR_Tr^Val and never varies its
size, although collecting group-labelled held-out data is the method's main practical cost.
This script applies the complete DFR_Tr^Val procedure of Appendix C to *shrunken* validation
sets - a random fraction n/133 of every group, so the composition is preserved and the minority
groups have n images - for n in {5, ..., 133}, using the penultimate-layer embeddings cached by
04_eval_table1.py for the Appendix-C base model.  Each n is repeated with several independent
random shrinkages.  The same sweep is run with an l2 instead of the paper's l1 penalty, to see
whether the sparsity-inducing penalty matters more when the reweighting set is small.

    python 08_size_sweep.py --npz results/table9/base_seed1_embeddings.npz --out_dir results/beyond \
        [--n_list 5,10,20,35,50,75,100,133] [--reps 10] [--refits 20] [--penalties l1,l2] [--jobs 8]

Procedure for one (n, repetition), following the authors' dfr_evaluate_spurious.py:
  1. available set = round(n/133 * group size) random images of each validation group;
  2. split it in half (stratified by group); group-balance the first half; standardise on it; fit
     the logistic regression on it for each C in {1, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01}; keep the C
     with the best worst-group accuracy on the second half;
  3. final last layer = average of `refits` (20, as in the authors' code) logistic regressions,
     each fit with that C on a fresh group-balanced draw (n per group) from the whole available
     set, standardised on that draw;
  4. evaluate on the test set: worst-group accuracy and training-weighted mean accuracy.
At n = 133 the available set is the full validation set and the procedure is exactly DFR_Tr^Val.
`--design balanced` instead uses exactly n images per group with a single final fit (an earlier,
simpler variant kept for comparison).

Outputs in --out_dir: size_sweep_runs.csv (one row per repetition), size_sweep_summary.csv,
size_sweep.tex (table), size_sweep.pdf / .png (figure).  Reference lines for the figure are read
from results/table9/base_seed1.json (ERM base model) and results/table9/table9_ours.json
(DFR_Tr^Val with the authors' script) when present.
"""
import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)  # sklearn >= 1.8 deprecation of `penalty`

C_GRID = [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]
GROUPS = ["landbird_on_land", "landbird_on_water", "waterbird_on_land", "waterbird_on_water"]  # g = 2*y + place


def load_sets(npz_path):
    d = np.load(npz_path)
    sets = {}
    for s in ("original_train", "original_val", "original_test"):
        sets[s] = dict(x=d[f"{s}_emb"].astype(np.float32), y=d[f"{s}_y"].astype(int), g=d[f"{s}_g"].astype(int))
    return sets


def group_accuracies(pred, y, g, n_groups=4):
    correct = pred == y
    return np.array([correct[g == k].mean() if (g == k).any() else np.nan for k in range(n_groups)])


def _sklearn_at_least(major, minor):
    import sklearn
    parts = sklearn.__version__.split(".")
    return (int(parts[0]), int(parts[1])) >= (major, minor)


def fit(x, y, C, penalty):
    # scikit-learn >= 1.8 replaces `penalty` by `l1_ratio` (1 = l1, 0 = l2); older versions use `penalty`.
    if _sklearn_at_least(1, 8):
        clf = LogisticRegression(l1_ratio=1.0 if penalty == "l1" else 0.0, C=C, solver="liblinear", max_iter=1000)
    else:
        clf = LogisticRegression(penalty=penalty, C=C, solver="liblinear", max_iter=1000)
    clf.fit(x, y)
    return clf


def balanced_draw(idx, g, n, rng):
    """n random indices from each group present in idx (all of a group if it has fewer than n)."""
    out = []
    for k in range(4):
        members = idx[g[idx] == k]
        if len(members) == 0:
            continue
        out.append(rng.choice(members, size=min(n, len(members)), replace=False))
    return np.concatenate(out)


def one_run(sets, n, rep, penalty, train_w, seed, refits=20, design="natural", n_ref=133):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # worker processes do not inherit the filters
    warnings.filterwarnings("ignore", category=FutureWarning)
    rng = np.random.RandomState(seed + 7919 * rep + 104729 * n + (0 if penalty == "l1" else 1))
    val, test = sets["original_val"], sets["original_test"]
    group_sizes = np.bincount(val["g"], minlength=4)

    # 1. available (shrunken) validation set, split in two halves stratified by group
    fit_idx, sel_idx = [], []
    for k in range(4):
        if design == "natural":
            take = int(round(n / n_ref * group_sizes[k])) if n < n_ref else int(group_sizes[k])
            take = max(take, 2)
        else:
            take = n
        pool = rng.permutation(np.where(val["g"] == k)[0])[:take]
        half = max(len(pool) // 2, 1)
        fit_idx.append(pool[:half])
        sel_idx.append(pool[half:] if len(pool) > 1 else pool[:1])
    fit_idx, sel_idx = np.concatenate(fit_idx), np.concatenate(sel_idx)
    avail_idx = np.unique(np.concatenate([fit_idx, sel_idx]))

    # 2. tune C: balance the fitting half, score worst-group accuracy on the other half
    fit_bal = balanced_draw(fit_idx, val["g"], int(np.bincount(val["g"][fit_idx], minlength=4)[np.bincount(val["g"][fit_idx], minlength=4) > 0].min()), rng)
    sc = StandardScaler().fit(val["x"][fit_bal])
    xf, xs = sc.transform(val["x"][fit_bal]), sc.transform(val["x"][sel_idx])
    best_c, best_wg = None, -1.0
    for C in C_GRID:
        clf = fit(xf, val["y"][fit_bal], C, penalty)
        wg = np.nanmin(group_accuracies(clf.predict(xs), val["y"][sel_idx], val["g"][sel_idx]))
        if wg > best_wg:
            best_c, best_wg = C, wg

    # 3. final last layer: average the coefficients of `refits` fits on fresh balanced draws of n per group
    n_bal = int(np.bincount(val["g"][avail_idx], minlength=4).min())
    n_fits = refits if design == "natural" else 1
    coefs, intercepts, nnz = [], [], []
    for _ in range(n_fits):
        draw = balanced_draw(avail_idx, val["g"], n_bal, rng)
        sc = StandardScaler().fit(val["x"][draw])
        clf = fit(sc.transform(val["x"][draw]), val["y"][draw], best_c, penalty)
        # fold the standardisation into the weights so the averaged layer acts on raw features
        w = clf.coef_[0] / sc.scale_
        b = clf.intercept_[0] - np.dot(w, sc.mean_)
        coefs.append(w); intercepts.append(b); nnz.append(int((clf.coef_ != 0).sum()))
    w_avg, b_avg = np.mean(coefs, axis=0), float(np.mean(intercepts))

    # 4. evaluate on the test set
    pred = (test["x"] @ w_avg + b_avg > 0).astype(int)
    accs = group_accuracies(pred, test["y"], test["g"])
    row = dict(penalty=penalty, n_per_group=n, rep=rep, n_available=int(len(avail_idx)), n_balanced=n_bal,
               refits=n_fits, best_C=best_c, tune_worst=100 * best_wg,
               worst=100 * accs.min(), adj_mean=100 * float((train_w * accs).sum()),
               n_nonzero=float(np.mean(nnz)))
    row.update({f"acc_{name}": 100 * a for name, a in zip(GROUPS, accs)})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out_dir", default="results/beyond")
    p.add_argument("--n_list", default="5,10,20,35,50,75,100,133")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--penalties", default="l1,l2")
    p.add_argument("--refits", type=int, default=20, help="averaged final fits (authors' code: 20)")
    p.add_argument("--design", choices=["natural", "balanced"], default="natural",
                   help="natural: shrink the validation set keeping its composition (default); balanced: exactly n per group, single fit")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--table9_dir", default="results/table9", help="for the ERM / full-DFR reference lines")
    p.add_argument("--ref_lines", default="erm", help="which dashed reference lines to draw: 'erm', 'erm,dfr' or 'none'")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    sets = load_sets(args.npz)
    counts = np.bincount(sets["original_train"]["g"], minlength=4)
    train_w = counts / counts.sum()
    max_n = int(np.bincount(sets["original_val"]["g"], minlength=4).min())
    n_list = [n for n in (int(v) for v in args.n_list.split(",")) if n <= max_n]
    penalties = args.penalties.split(",")
    print(f"val group sizes allow n <= {max_n}; running n = {n_list}, {args.reps} reps, penalties {penalties}")

    jobs = [(n, r, pen) for pen in penalties for n in n_list for r in range(args.reps)]
    print(f"design = {args.design}, {args.refits if args.design == 'natural' else 1} averaged final fit(s)")
    rows = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(one_run)(sets, n, r, pen, train_w, args.seed, args.refits, args.design, max_n) for n, r, pen in jobs)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(args.out_dir, "size_sweep_runs.csv"), index=False)

    summary = (runs.groupby(["penalty", "n_per_group"])
               .agg(n_available=("n_available", "first"), worst_mean=("worst", "mean"), worst_std=("worst", "std"),
                    adj_mean_mean=("adj_mean", "mean"), adj_mean_std=("adj_mean", "std"),
                    best_C_median=("best_C", "median"), n_nonzero_mean=("n_nonzero", "mean"), reps=("rep", "count"))
               .reset_index())
    summary.to_csv(os.path.join(args.out_dir, "size_sweep_summary.csv"), index=False)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    # reference numbers from the Table 9 run, if available
    refs = {}
    base_json = os.path.join(args.table9_dir, "base_seed1.json")
    t9_json = os.path.join(args.table9_dir, "table9_ours.json")
    if os.path.exists(base_json):
        b = json.load(open(base_json))["sets"]["original_test"]
        refs["ERM base model"] = (b["worst_group_acc"], b["adjusted_mean_acc"])
    if os.path.exists(t9_json):
        dv = json.load(open(t9_json)).get("dfr_val", {})
        if dv:
            first = dv[sorted(dv)[0]]
            refs["DFR (authors' script, all 133 per group)"] = (first["worst"], first["mean_adjusted"])
    json.dump(refs, open(os.path.join(args.out_dir, "size_sweep_refs.json"), "w"), indent=2)

    # ---- LaTeX table ----
    lines = [r"\begin{table}[h]", r"\centering", r"\begin{tabular}{r " + "cc " * len(penalties) + "}", r"\hline",
             "Examples per group $n$ & " + " & ".join(
                 rf"\multicolumn{{2}}{{c}}{{$\ell_{pen[1]}$ penalty}}" for pen in penalties) + r" \\",
             " & " + " & ".join("Worst (\\%) & Mean (\\%)" for _ in penalties) + r" \\", r"\hline"]
    for n in n_list:
        cells = []
        for pen in penalties:
            s = summary[(summary.penalty == pen) & (summary.n_per_group == n)].iloc[0]
            cells += [f"{s.worst_mean:.1f} $\\pm$ {s.worst_std:.1f}", f"{s.adj_mean_mean:.1f} $\\pm$ {s.adj_mean_std:.1f}"]
        n_av = int(summary[summary.n_per_group == n].n_available.iloc[0])
        lines.append(f"{n} ({n_av} images available) & " + " & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}",
              rf"\caption{{Worst-group and training-weighted mean test accuracy of $\mathrm{{DFR}}^{{\mathrm{{Val}}}}_{{\mathrm{{Tr}}}}$ on "
              rf"Waterbirds as a function of the size of the group-labelled validation set, shrunk at random to $n$ minority-group "
              rf"images per group while keeping its composition; the full Appendix~C procedure (C tuned on half, {args.refits} averaged refits) "
              rf"is run on each shrunken set. Mean $\pm$ std over {args.reps} random shrinkages (ours; the paper uses the full set, $n=133$). "
              + (rf"ERM base model: {refs['ERM base model'][0]:.1f} / {refs['ERM base model'][1]:.1f}." if "ERM base model" in refs else "")
              + "}", r"\label{tab:size_sweep}", r"\end{table}"]
    open(os.path.join(args.out_dir, "size_sweep.tex"), "w").write("\n".join(lines) + "\n")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"l1": "#2a78d6", "l2": "#eb6834"}
    markers = {"l1": "o", "l2": "s"}
    labels = {"l1": r"DFR, $\ell_1$ penalty (paper's setting)", "l2": r"DFR, $\ell_2$ penalty"}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True)
    for ax, (col, err, ylabel, ref_i) in zip(axes, [("worst_mean", "worst_std", "Worst-group test accuracy (%)", 0),
                                                    ("adj_mean_mean", "adj_mean_std", "Mean test accuracy (%)", 1)]):
        for pen in penalties:
            s = summary[summary.penalty == pen].sort_values("n_per_group")
            ax.errorbar(s.n_per_group, s[col], yerr=s[err], color=colors.get(pen, "#444444"), marker=markers.get(pen, "o"),
                        ms=5, lw=1.6, capsize=2.5, elinewidth=1, label=labels.get(pen, pen))
        wanted = args.ref_lines.split(",")
        shown = {k: v for k, v in refs.items() if ("erm" in wanted and k.startswith("ERM")) or ("dfr" in wanted and k.startswith("DFR"))}
        for j, (name, (w, m)) in enumerate(shown.items()):
            v = (w, m)[ref_i]
            ax.axhline(v, color="#6b6b6b", lw=1, ls="--", zorder=0)
            ax.annotate(f"{name}: {v:.1f}", xy=(n_list[-1], v), xytext=(0, 3 if j % 2 == 0 else -10),
                        textcoords="offset points", ha="right", fontsize=7.5, color="#52514e")
        ax.set_xscale("log")
        ax.set_xlim(n_list[0] / 1.35, n_list[-1] * 1.2)  # pin the range; some matplotlib versions autoscale log axes wide
        ax.set_xticks(n_list)
        ax.set_xticklabels([str(n) for n in n_list], fontsize=8)
        ax.minorticks_off()
        ax.set_xlabel("Minority-group images $n$ in the shrunken validation set", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", color="#e3e3e0", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#9a9a96")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, fontsize=8, frameon=False, loc="lower center", ncol=len(penalties),
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out_dir, f"size_sweep.{ext}"), dpi=200, bbox_inches="tight")
    print(f"\nwrote size_sweep_runs.csv, size_sweep_summary.csv, size_sweep.tex, size_sweep.pdf/.png in {args.out_dir}")


if __name__ == "__main__":
    main()
