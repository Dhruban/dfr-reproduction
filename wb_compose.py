"""Shared image-compositing helpers for the Waterbirds variants.

`crop_and_resize` and `combine_and_mask` follow `dataset_scripts/dataset_utils.py` in
github.com/kohpangwei/group_DRO (the code the Waterbirds tarball was generated with, and the
code the DFR authors' `generate_waterbirds_fg_bg.ipynb` imports), with two changes:
`Image.ANTIALIAS` -> `Image.LANCZOS` (removed in Pillow 10) and the final add is clipped to
255 instead of wrapping around in uint8.
"""
import os

import numpy as np
from PIL import Image

GREY = 150  # FG-Only background value used in the authors' notebook: np.ones_like(place) * 150

# Places365 categories used by Waterbirds; index 0 = land, 1 = water (same as `place` in metadata.csv)
PLACE_CATEGORIES = {
    0: ["/b/bamboo_forest", "/f/forest/broadleaf"],
    1: ["/o/ocean", "/l/lake/natural"],
}


def crop_and_resize(source_img, target_img):
    """Resize/crop `source_img` so that it has exactly the size of `target_img`."""
    source_width, source_height = source_img.size
    target_width, target_height = target_img.size

    # Source does not completely cover target: enlarge it first, then crop.
    if (source_width < target_width) or (source_height < target_height):
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.LANCZOS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.LANCZOS)
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:  # crop left/right
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        box = (offset, 0, source_width - offset, source_height)
    else:  # crop top/bottom
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        box = (0, offset, source_width, source_height - offset)
    return source_img.crop(box).resize((target_width, target_height), Image.LANCZOS)


def combine_and_mask(img_new, mask, img_black):
    """Paste the masked bird (`img_black`, bird on black) onto `img_new` using `mask` in [0, 1]."""
    img_resized = crop_and_resize(img_new, img_black)
    img_resized_np = np.asarray(img_resized).astype(np.float64)
    img_masked_np = np.around(img_resized_np * (1 - mask))
    combined = np.asarray(img_black).astype(np.int16) + img_masked_np.astype(np.int16)
    return Image.fromarray(np.clip(combined, 0, 255).astype(np.uint8))


def load_bird(cub_dir, img_filename):
    """Return (mask in [0,1] as HxWx3 float, bird-on-black PIL image) for a CUB image."""
    img_path = os.path.join(cub_dir, "images", img_filename)
    seg_path = os.path.join(cub_dir, "segmentations", img_filename.replace(".jpg", ".png"))
    img = Image.open(img_path).convert("RGB")
    seg = Image.open(seg_path).convert("RGB")
    if seg.size != img.size:  # never happens in CUB as shipped, but be safe
        seg = seg.resize(img.size, Image.NEAREST)
    img_np = np.asarray(img).astype(np.float64)
    seg_np = np.asarray(seg).astype(np.float64) / 255.0
    img_black = Image.fromarray(np.around(img_np * seg_np).astype(np.uint8))
    return seg_np, img_black


def compose_on_place(cub_dir, img_filename, place_path):
    """Original-style Waterbirds image: bird pasted onto a Places photo."""
    seg_np, img_black = load_bird(cub_dir, img_filename)
    place = Image.open(place_path).convert("RGB")
    return combine_and_mask(place, seg_np, img_black)


def compose_fg_only(cub_dir, img_filename, grey=GREY):
    """FG-Only Waterbirds image: bird pasted onto a uniform grey background."""
    seg_np, img_black = load_bird(cub_dir, img_filename)
    grey_img = Image.new("RGB", img_black.size, (grey, grey, grey))
    return combine_and_mask(grey_img, seg_np, img_black)


def save_jpeg(img, path):
    """Save like the authors' notebook did (PIL defaults: JPEG quality 75)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def list_place_images(places_dir, place_type):
    """All Places images of the given type (0 land / 1 water) as '/x/category/file.jpg' strings."""
    files = []
    for cat in PLACE_CATEGORIES[place_type]:
        cat_dir = os.path.join(places_dir, cat.lstrip("/"))
        if not os.path.isdir(cat_dir):
            raise FileNotFoundError(f"Places category folder not found: {cat_dir}")
        files += [f"{cat}/{f}" for f in sorted(os.listdir(cat_dir)) if f.lower().endswith(".jpg")]
    return files
