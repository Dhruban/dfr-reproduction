"""Evaluate trained Waterbirds checkpoints on the Original and FG-Only test sets.

    # every run under logs/table1 that has a final_checkpoint.pt
    python 04_eval_table1.py --repo_dir $REPO --runs_dir logs/table1 \
        --wb_dir $DATA_ROOT/waterbirds --fg_dir $DATA_ROOT/waterbirds_fg --out_dir results/table1

    # a single checkpoint
    python 04_eval_table1.py --repo_dir $REPO --ckpt logs/table1/original95_seed1/final_checkpoint.pt \
        --name original95_seed1 --wb_dir ... --fg_dir ... --out_dir results/table1

For each checkpoint and each test set this writes per-group accuracies, worst-group accuracy
(min over the 4 (y, place) groups, which is what the authors' code computes, and also min over
the 2 classes), the plain test-set mean, and the Sagawa-style "adjusted" mean that weights the
groups by their proportions in the standard training set (the Mean(%) column of the paper's
Table 2).  With --save_embeddings it also caches penultimate-layer (2048-d) features of the
Original train/val/test sets and the FG-Only test set, for the DFR experiments.

Uses the authors' dataset class and test-time transform (wb_data.py) so preprocessing is
identical to training.  Runs on GPU if available, otherwise CPU (slow but fine for a few runs).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torchvision
from tqdm import tqdm

GROUP_NAMES = {0: "landbird_on_land", 1: "landbird_on_water", 2: "waterbird_on_land", 3: "waterbird_on_water"}


def build_model(ckpt_path, n_classes=2, device="cpu"):
    model = torchvision.models.resnet50()
    model.fc = torch.nn.Linear(model.fc.in_features, n_classes)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


@torch.no_grad()
def run(model, loader, device, want_embeddings=False):
    """Return predictions, labels, groups, places (and embeddings) for a loader."""
    preds, ys, gs, ps, embs = [], [], [], [], []
    feat = torch.nn.Sequential(*list(model.children())[:-1])  # everything up to avgpool
    for x, y, g, p in tqdm(loader, leave=False):
        x = x.to(device)
        h = torch.flatten(feat(x), 1)
        logits = model.fc(h)
        preds.append(logits.argmax(1).cpu())
        ys.append(y); gs.append(g); ps.append(p)
        if want_embeddings:
            embs.append(h.cpu().numpy())
    out = dict(pred=torch.cat(preds).numpy(), y=torch.cat(ys).numpy(),
               g=torch.cat(gs).numpy(), p=torch.cat(ps).numpy())
    if want_embeddings:
        out["emb"] = np.concatenate(embs)
    return out


def summarize(res, train_group_frac):
    correct = res["pred"] == res["y"]
    n_groups = int(res["g"].max()) + 1
    group_acc = {int(g): float(correct[res["g"] == g].mean()) for g in range(n_groups)}
    group_n = {int(g): int((res["g"] == g).sum()) for g in range(n_groups)}
    class_acc = {int(c): float(correct[res["y"] == c].mean()) for c in np.unique(res["y"])}
    adjusted = float(sum(train_group_frac[g] * group_acc[g] for g in group_acc))
    return {
        "group_acc": {GROUP_NAMES.get(g, str(g)): round(100 * a, 2) for g, a in group_acc.items()},
        "group_n": {GROUP_NAMES.get(g, str(g)): n for g, n in group_n.items()},
        "worst_group_acc": round(100 * min(group_acc.values()), 2),
        "worst_class_acc": round(100 * min(class_acc.values()), 2),
        "mean_acc": round(100 * float(correct.mean()), 2),
        "adjusted_mean_acc": round(100 * adjusted, 2),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_dir", required=True, help="authors' deep_feature_reweighting repo (for wb_data.py)")
    p.add_argument("--wb_dir", required=True)
    p.add_argument("--fg_dir", required=True)
    p.add_argument("--out_dir", default="results/table1")
    p.add_argument("--runs_dir", default=None, help="evaluate every <runs_dir>/*/final_checkpoint.pt")
    p.add_argument("--ckpt", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save_embeddings", action="store_true")
    args = p.parse_args()

    sys.path.insert(0, args.repo_dir)
    from wb_data import WaterBirdsDataset, get_transform_cub  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.runs_dir:
        runs = sorted((d, os.path.join(args.runs_dir, d, "final_checkpoint.pt")) for d in os.listdir(args.runs_dir)
                      if os.path.exists(os.path.join(args.runs_dir, d, "final_checkpoint.pt")))
    else:
        assert args.ckpt and args.name, "give --runs_dir, or --ckpt together with --name"
        runs = [(args.name, args.ckpt)]
    print(f"{len(runs)} checkpoint(s): {[r[0] for r in runs]}")

    tf = get_transform_cub(target_resolution=(224, 224), train=False, augment_data=False)
    loader_kw = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device == "cuda")

    def make_loader(basedir, split):
        return torch.utils.data.DataLoader(WaterBirdsDataset(basedir=basedir, split=split, transform=tf), **loader_kw)

    # group proportions of the standard training set, for the "adjusted" mean accuracy
    train_ds = WaterBirdsDataset(basedir=args.wb_dir, split="train", transform=tf)
    train_group_frac = {int(g): float(c) / len(train_ds) for g, c in enumerate(train_ds.group_counts.numpy())}
    print("standard training group fractions:", {GROUP_NAMES[g]: round(f, 3) for g, f in train_group_frac.items()})

    eval_sets = {"original_test": (args.wb_dir, "test"), "fg_only_test": (args.fg_dir, "test"),
                 "original_val": (args.wb_dir, "val")}
    emb_sets = {"original_train": (args.wb_dir, "train"), "original_val": (args.wb_dir, "val"),
                "original_test": (args.wb_dir, "test"), "fg_only_test": (args.fg_dir, "test")}

    for name, ckpt in runs:
        t0 = time.time()
        model = build_model(ckpt, n_classes=train_ds.n_classes, device=device)
        result = {"name": name, "checkpoint": os.path.abspath(ckpt), "sets": {}}
        cache = {}
        todo = dict(eval_sets)
        if args.save_embeddings:
            todo.update(emb_sets)
        for set_name, (basedir, split) in todo.items():
            res = run(model, make_loader(basedir, split), device, want_embeddings=args.save_embeddings)
            cache[set_name] = res
            if set_name in eval_sets:
                result["sets"][set_name] = summarize(res, train_group_frac)
        result["eval_seconds"] = round(time.time() - t0, 1)

        with open(os.path.join(args.out_dir, f"{name}.json"), "w") as f:
            json.dump(result, f, indent=2)
        if args.save_embeddings:
            np.savez_compressed(os.path.join(args.out_dir, f"{name}_embeddings.npz"),
                                **{f"{k}_{field}": v for k, r in cache.items() for field, v in r.items()
                                   if field in ("emb", "y", "g", "p")})
        o, fg = result["sets"]["original_test"], result["sets"]["fg_only_test"]
        print(f"{name}: worst-group  Original {o['worst_group_acc']:.1f}%  FG-Only {fg['worst_group_acc']:.1f}%   "
              f"| mean Original {o['mean_acc']:.1f}% (adjusted {o['adjusted_mean_acc']:.1f}%)  "
              f"[{result['eval_seconds']} s]")


if __name__ == "__main__":
    main()
