"""Build the Balanced (50%) Waterbirds training set of Table 1, following Appendix B.1 of
Kirichenko et al.: start from the Original (95%) training data and replace the background of
the smallest possible number of randomly chosen training images so that, within each class,
half of the birds are on land and half on water.  Validation and test images are unchanged
(they are already balanced) and are linked from the standard directory.

    python 02_make_balanced.py --wb_dir DATA/waterbirds --cub_dir DATA/CUB_200_2011 \
        --places_dir DATA/places/data_256 --out_dir DATA/waterbirds_balanced --seed 0 [--copy]

New backgrounds are drawn without replacement from the Places images of the target category
pair that are NOT already used anywhere in the standard dataset (via `place_filename`).
"""
import argparse
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb_compose import compose_on_place, list_place_images, save_jpeg  # noqa: E402


def _compose_one(args):
    cub_dir, places_dir, out_dir, img_filename, place_filename = args
    out_path = os.path.join(out_dir, img_filename)
    place_path = os.path.join(places_dir, place_filename.lstrip("/"))
    save_jpeg(compose_on_place(cub_dir, img_filename, place_path), out_path)
    return img_filename


def group_table(df):
    t = df.groupby(["y", "place"]).size().unstack(fill_value=0)
    t.index = ["landbird (y=0)", "waterbird (y=1)"][: len(t.index)]
    t.columns = [f"place={c} ({'land' if c == 0 else 'water'})" for c in t.columns]
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wb_dir", required=True)
    p.add_argument("--cub_dir", required=True)
    p.add_argument("--places_dir", required=True, help="e.g. DATA/places/data_256 (contains b/, f/, o/, l/)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--copy", action="store_true", help="copy unchanged images instead of symlinking them")
    args = p.parse_args()
    rng = np.random.RandomState(args.seed)

    meta = pd.read_csv(os.path.join(args.wb_dir, "metadata.csv"))
    has_pf = "place_filename" in meta.columns
    if not has_pf:
        print("WARNING: metadata.csv has no place_filename column; cannot exclude already-used Places "
              "images, drawing from all images of each category instead.")
        meta["place_filename"] = ""
    train = meta["split"] == 0
    print("Original training groups:\n", group_table(meta[train]), "\n")

    # ---- choose the minimal random set of training images to flip ------------------------------
    flip_idx = []
    for y in (0, 1):
        cls = meta.index[train & (meta["y"] == y)]
        n_by_place = meta.loc[cls, "place"].value_counts()
        maj = int(n_by_place.idxmax())
        n_flip = int((n_by_place.max() - n_by_place.min()) // 2)
        cand = meta.index[train & (meta["y"] == y) & (meta["place"] == maj)].to_numpy()
        chosen = rng.choice(cand, size=n_flip, replace=False)
        flip_idx.append(chosen)
        print(f"class y={y}: majority place={maj} ({n_by_place.max()}), minority ({n_by_place.min()}); "
              f"flipping {n_flip} images from place {maj} to {1 - maj}")
    flip_idx = np.concatenate(flip_idx)

    new = meta.copy()
    new["flipped"] = 0
    new.loc[flip_idx, "flipped"] = 1
    new.loc[flip_idx, "place"] = 1 - new.loc[flip_idx, "place"]

    # ---- assign fresh Places backgrounds ------------------------------------------------------
    used = set(meta["place_filename"].astype(str)) if has_pf else set()
    for place_type in (0, 1):
        rows = new.index[(new["flipped"] == 1) & (new["place"] == place_type)]
        pool = [f for f in list_place_images(args.places_dir, place_type) if f not in used]
        print(f"place {place_type}: need {len(rows)} new backgrounds, {len(pool)} unused Places images available")
        if len(pool) >= len(rows):
            pick = rng.choice(pool, size=len(rows), replace=False)
        else:
            print("  WARNING: not enough unused images, sampling WITH replacement from the unused pool")
            pick = rng.choice(pool, size=len(rows), replace=True)
        new.loc[rows, "place_filename"] = pick

    print("\nBalanced training groups:\n", group_table(new[train]), "\n")

    # ---- write images -------------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(args.cub_dir, args.places_dir, args.out_dir, r.img_filename, r.place_filename)
            for r in new.loc[flip_idx].itertuples()]
    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(_compose_one, jobs, chunksize=16), total=len(jobs),
                      desc="re-backgrounding"):
            pass

    keep = new.index[new["flipped"] == 0]
    for f in tqdm(new.loc[keep, "img_filename"], desc="linking unchanged images"):
        src = os.path.abspath(os.path.join(args.wb_dir, f))
        dst = os.path.join(args.out_dir, f)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            continue
        if args.copy:
            shutil.copy(src, dst)
        else:
            os.symlink(src, dst)

    cols = [c for c in meta.columns]  # same columns as the standard metadata.csv
    new[cols].to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)
    new.loc[flip_idx, ["img_filename", "y", "place", "place_filename"]].to_csv(
        os.path.join(args.out_dir, "balanced_flips.csv"), index=False)
    print(f"wrote {args.out_dir}/metadata.csv ({len(flip_idx)} training images re-backgrounded, "
          f"list in balanced_flips.csv)")


if __name__ == "__main__":
    main()
