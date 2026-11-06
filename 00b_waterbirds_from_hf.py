"""Rebuild the standard Waterbirds folder (images in CUB class folders + metadata.csv) from the
Hugging Face mirror `grodino/waterbirds`, for when nlp.stanford.edu is unreachable.

    python 00b_waterbirds_from_hf.py --hf_dir DATA/hf_waterbirds --out_dir DATA/waterbirds [--cub_dir DATA/CUB_200_2011]

The mirror stores, per image, the original JPEG bytes, the original CUB file name (`image.path`),
the species index (`bird`, 0-199 in CUB order), `label` (y) and `place`; the split is the parquet
file name.  We write each image unchanged to <out_dir>/<NNN.Species>/<file>.jpg and produce
metadata.csv with the standard columns (img_id, img_filename, y, split, place).  The original
`place_filename` column is not in the mirror, so it is omitted; the other scripts cope with that.
If --cub_dir is given and CUB is extracted, every reconstructed path is checked against CUB's
images.txt and img_id is taken from it.  Group counts are compared with the known counts of the
standard release and any mismatch is reported.
"""
import argparse
import glob
import json
import os

import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

SPLITS = {"train": 0, "validation": 1, "val": 1, "test": 2}
# (split, y, place) -> count in waterbird_complete95_forest2water2
EXPECTED = {
    (0, 0, 0): 3498, (0, 0, 1): 184, (0, 1, 0): 56, (0, 1, 1): 1057,
    (1, 0, 0): 467, (1, 0, 1): 466, (1, 1, 0): 133, (1, 1, 1): 133,
    (2, 0, 0): 2255, (2, 0, 1): 2255, (2, 1, 0): 642, (2, 1, 1): 642,
}


def species_names(table):
    """Class names of the `bird` column from the Hugging Face schema metadata."""
    meta = json.loads(table.schema.metadata[b"huggingface"].decode())
    feat = meta["info"]["features"]["bird"]
    assert feat.get("_type") == "ClassLabel", "bird column is not a ClassLabel"
    return feat["names"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hf_dir", required=True, help="folder with data/{train,validation,test}-*.parquet")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--cub_dir", default=None, help="optional: CUB_200_2011 dir to verify paths / get img_id")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.hf_dir, "**", "*.parquet"), recursive=True))
    assert files, f"no parquet files under {args.hf_dir}"

    # CUB's images.txt, when available, gives the authoritative "<NNN.Species>/<file>.jpg" path for
    # every file name (file names are unique in CUB) and the original img_id.
    base2path, path2id = {}, {}
    images_txt = os.path.join(args.cub_dir, "images.txt") if args.cub_dir else None
    if images_txt and os.path.exists(images_txt):
        cub = pd.read_csv(images_txt, sep=" ", header=None, names=["img_id", "img_filename"])
        base2path = {os.path.basename(p): p for p in cub["img_filename"]}
        path2id = dict(zip(cub["img_filename"], cub["img_id"]))
        print(f"using CUB images.txt ({len(cub)} files) to resolve paths")
    else:
        print("CUB images.txt not available; deriving class folders from the mirror's species names")

    rows = []
    names = None
    n_unresolved = 0
    for f in files:
        split_name = os.path.basename(f).split("-")[0]
        assert split_name in SPLITS, f"cannot infer split from file name {f}"
        split = SPLITS[split_name]
        t = pq.read_table(f)
        names = names or species_names(t)
        n_written = 0
        for batch in tqdm(t.to_batches(max_chunksize=256), desc=f"{split_name:<10}", total=(t.num_rows + 255) // 256):
            for r in batch.to_pylist():
                img = r["image"]
                base = os.path.basename(img["path"])
                derived = f"{r['bird'] + 1:03d}.{names[r['bird']]}/{base}"
                img_filename = base2path.get(base, derived)
                n_unresolved += base2path != {} and base not in base2path
                out = os.path.join(args.out_dir, img_filename)
                if not os.path.exists(out):
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out, "wb") as fh:
                        fh.write(img["bytes"])
                    n_written += 1
                rows.append(dict(img_id=int(path2id.get(img_filename, -1)), img_filename=img_filename,
                                 y=int(r["label"]), split=split, place=int(r["place"])))
        print(f"{split_name}: {t.num_rows} rows, {n_written} images written")
    if base2path:
        print(f"file names not found in CUB images.txt: {n_unresolved}" + ("  (all good)" if n_unresolved == 0 else "  <-- check cub_dir"))

    meta = pd.DataFrame(rows)
    assert meta["img_filename"].is_unique, "duplicate file names in the mirror"
    if (meta["img_id"] < 0).any():
        meta["img_id"] = range(1, len(meta) + 1)  # running index; img_id is not used by any script
    meta = meta[["img_id", "img_filename", "y", "split", "place"]].sort_values("img_id").reset_index(drop=True)
    os.makedirs(args.out_dir, exist_ok=True)
    meta.to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)

    # compare group counts with the standard release
    counts = meta.groupby(["split", "y", "place"]).size().to_dict()
    bad = 0
    print(f"\n{'split':>6} {'y':>2} {'place':>5} {'ours':>6} {'standard':>9}")
    for k in sorted(EXPECTED):
        ours, exp = counts.get(k, 0), EXPECTED[k]
        bad += ours != exp
        print(f"{k[0]:>6} {k[1]:>2} {k[2]:>5} {ours:>6} {exp:>9}" + ("" if ours == exp else "   <-- differs"))
    print(f"\ntotal {len(meta)} images (standard: 11788); {bad} group counts differ")
    print(f"wrote {args.out_dir}/metadata.csv  columns={list(meta.columns)}")


if __name__ == "__main__":
    main()
