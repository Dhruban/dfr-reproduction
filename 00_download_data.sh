#!/bin/bash
# Download everything needed for Table 1 of Kirichenko et al. (ICLR 2023).
#
#   bash 00_download_data.sh <DATA_ROOT> [small|large|none]
#
#   DATA_ROOT  : where to put the data (use /scratch on Great Lakes, not $HOME)
#   Places size: small  = 256x256 Places365 train set (24 GB download)   [default]
#                large  = full-resolution Places365 train set (105 GB)    <- what Sagawa et al. used
#                none   = skip Places (enough for the FG-Only row and for the two Original rows)
#
# Places is needed ONLY for the Balanced (50%) training set (02_make_balanced.py) and for the
# optional pixel-level alignment check in 01_make_fg_only.py.  Only the four Waterbirds
# background categories are extracted from the tar; the tar itself is deleted afterwards.
#
# Run this as a batch job (standard partition, 1 CPU, ~6 h) or under nohup on a login node;
# the Places download is the slow part.
set -euo pipefail

DATA_ROOT=${1:?usage: bash 00_download_data.sh <DATA_ROOT> [small|large|none]}
PLACES=${2:-small}
mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"
echo "[$(date)] data root: $DATA_ROOT"

# ------------------------------------------------------------------------------------------
# 1. Standard Waterbirds (Sagawa et al. 2020), the pre-generated 95% tarball used by everyone.
# ------------------------------------------------------------------------------------------
if [ ! -f waterbirds/metadata.csv ]; then
  echo "[$(date)] downloading Waterbirds tarball"
  wget -c https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
  mkdir -p waterbirds_extract
  tar -xzf waterbird_complete95_forest2water2.tar.gz -C waterbirds_extract
  META=$(find waterbirds_extract -maxdepth 2 -name metadata.csv | head -1)
  [ -n "$META" ] || { echo "metadata.csv not found after extracting the Waterbirds tarball"; exit 1; }
  mv "$(dirname "$META")" waterbirds
  rm -rf waterbirds_extract
fi
echo "[$(date)] Waterbirds: $(wc -l < waterbirds/metadata.csv) metadata rows; header: $(head -1 waterbirds/metadata.csv)"

# ------------------------------------------------------------------------------------------
# 2. CUB-200-2011 images (1.1 GB) and the segmentation masks (39 MB) -> CUB_200_2011/{images,segmentations}
# ------------------------------------------------------------------------------------------
if [ ! -d CUB_200_2011/images ]; then
  echo "[$(date)] downloading CUB-200-2011"
  wget -c https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
  tar -xzf CUB_200_2011.tgz            # -> CUB_200_2011/ (+ a stray attributes.txt)
fi
if [ ! -d CUB_200_2011/segmentations ]; then
  echo "[$(date)] downloading CUB segmentations"
  wget -c -O segmentations.tgz "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz?download=1"
  tar -xzf segmentations.tgz -C CUB_200_2011   # -> CUB_200_2011/segmentations/
fi
echo "[$(date)] CUB images: $(find CUB_200_2011/images -name '*.jpg' | wc -l), masks: $(find CUB_200_2011/segmentations -name '*.png' | wc -l)"

# ------------------------------------------------------------------------------------------
# 3. Places365: only the 4 background categories used by Waterbirds.
#    land  = /b/bamboo_forest, /f/forest/broadleaf ; water = /o/ocean, /l/lake/natural
# ------------------------------------------------------------------------------------------
case "$PLACES" in
  none)
    echo "[$(date)] skipping Places" ;;
  small|large)
    if [ "$PLACES" = small ]; then
      TAR=train_256_places365standard.tar; SUB=data_256
    else
      TAR=train_large_places365standard.tar; SUB=data_large
    fi
    mkdir -p places
    if [ ! -d places/$SUB/o/ocean ]; then
      echo "[$(date)] downloading Places365 ($PLACES) - this is the slow step"
      wget -c http://data.csail.mit.edu/places/places365/$TAR
      echo "[$(date)] extracting the 4 Waterbirds categories (one pass over the whole tar)"
      tar -xf $TAR -C places --wildcards \
        "*$SUB/b/bamboo_forest/*" "*$SUB/f/forest/broadleaf/*" "*$SUB/o/ocean/*" "*$SUB/l/lake/natural/*"
      rm -f $TAR
    fi
    for c in b/bamboo_forest f/forest/broadleaf o/ocean l/lake/natural; do
      echo "   $c: $(ls places/$SUB/$c | wc -l) images"
    done
    echo "[$(date)] Places dir to pass to the scripts: $DATA_ROOT/places/$SUB" ;;
  *)
    echo "unknown Places option '$PLACES' (use small, large or none)"; exit 1 ;;
esac

echo "[$(date)] done"
