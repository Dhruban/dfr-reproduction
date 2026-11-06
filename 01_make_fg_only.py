"""Build the FG-Only Waterbirds dataset (birds on a uniform grey background), aligned 1:1 with
the standard Waterbirds tarball: same images, same metadata.csv, same train/val/test split.

    python 01_make_fg_only.py --wb_dir DATA/waterbirds --cub_dir DATA/CUB_200_2011 \
        --out_dir DATA/waterbirds_fg --workers 8 \
        [--places_dir DATA/places/data_256 --n_check 50]

Needs only CUB images + CUB segmentation masks.  If --places_dir is given, the script also
re-composites `n_check` random training images onto their original Places backgrounds (taken
from `place_filename` in metadata.csv) and reports the pixel difference to the tarball images,
as a check that the compositing code reproduces the standard data.
"""
import argparse
import os
import shutil
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wb_compose import compose_fg_only, compose_on_place, save_jpeg  # noqa: E402


def _make_one(args):
    cub_dir, out_dir, img_filename = args
    out_path = os.path.join(out_dir, img_filename)
    if os.path.exists(out_path):
        return img_filename, "exists"
    save_jpeg(compose_fg_only(cub_dir, img_filename), out_path)
    return img_filename, "ok"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wb_dir", required=True, help="standard Waterbirds dir (contains metadata.csv)")
    p.add_argument("--cub_dir", required=True, help="CUB_200_2011 dir (contains images/ and segmentations/)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--places_dir", default=None, help="optional: Places dir for the alignment check")
    p.add_argument("--n_check", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    meta = pd.read_csv(os.path.join(args.wb_dir, "metadata.csv"))
    print(f"{len(meta)} images; columns: {list(meta.columns)}")
    for col in ("img_filename", "y", "split", "place"):
        assert col in meta.columns, f"metadata.csv is missing column {col}"

    # Cheap alignment check on every image: the composite has the size of the CUB bird image,
    # so the tarball image and the CUB image must have identical sizes.
    mismatched = 0
    for f in tqdm(meta["img_filename"], desc="size check"):
        with Image.open(os.path.join(args.wb_dir, f)) as a, \
                Image.open(os.path.join(args.cub_dir, "images", f)) as b:
            mismatched += a.size != b.size
    print(f"size check: {mismatched} of {len(meta)} tarball images differ in size from the CUB image "
          f"({'OK' if mismatched == 0 else 'NOT aligned - check cub_dir'})")

    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(args.cub_dir, args.out_dir, f) for f in meta["img_filename"]]
    with Pool(args.workers) as pool:
        for _ in tqdm(pool.imap_unordered(_make_one, jobs, chunksize=32), total=len(jobs), desc="FG-Only"):
            pass
    shutil.copy(os.path.join(args.wb_dir, "metadata.csv"), os.path.join(args.out_dir, "metadata.csv"))
    print(f"wrote {len(meta)} FG-Only images + metadata.csv to {args.out_dir}")

    # Optional pixel-level check that bird + place_filename reproduces the tarball image.
    if args.places_dir and "place_filename" in meta.columns:
        rng = np.random.RandomState(args.seed)
        rows = meta[meta["split"] == 0].sample(n=min(args.n_check, (meta["split"] == 0).sum()),
                                              random_state=rng)
        diffs = []
        for _, r in tqdm(rows.iterrows(), total=len(rows), desc="alignment check"):
            place_path = os.path.join(args.places_dir, r["place_filename"].lstrip("/"))
            ours = np.asarray(compose_on_place(args.cub_dir, r["img_filename"], place_path)).astype(float)
            theirs = np.asarray(Image.open(os.path.join(args.wb_dir, r["img_filename"])).convert("RGB")).astype(float)
            diffs.append(np.abs(ours - theirs).mean())
        diffs = np.array(diffs)
        print(f"alignment check on {len(diffs)} train images: mean |pixel diff| = {diffs.mean():.2f} "
              f"(median {np.median(diffs):.2f}, max {diffs.max():.2f}) on a 0-255 scale.")
        print("  A few units is JPEG/resampling noise and means the metadata+compositing reproduce the "
              "tarball; tens of units would mean the Places images or resolution differ.")
    elif args.places_dir:
        print("metadata.csv has no place_filename column; skipping the pixel-level alignment check")


if __name__ == "__main__":
    main()
