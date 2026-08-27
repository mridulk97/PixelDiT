#!/usr/bin/env bash
#
# Put Distinctions-646 on disk in the layout Distinctions646MattingDataset
# expects, and keep it there.
#
#   <root>/Train/FG/<name>.png    596 foregrounds
#   <root>/Train/GT/<name>.png    596 alphas, 1:1 with FG
#   <root>/Train/bg_train.txt     59,600 COCO background names
#   <root>/Test/{FG,GT}            50 pairs
#   <root>/Test/bg_test.txt       1,000 VOC background names
#
# Composites are built on the fly by Distinctions646MattingDataset, so the
# 59,600 PNGs (~100 GB) gen_train.py would write never land on disk.
#
# Idempotent, like setup_am2k_data.sh: run it before every session and it only
# re-stamps timestamps when the data is already complete.
#
# TIMESTAMPS
#
# Same trap as AM-2K: archives carry their original build dates, so a plain
# extract can land files that an age-based scratch cleanup treats as years
# stale and deletes on its next sweep, while the archive itself survives. This
# script touches everything it extracts. Do not drop that step.
#
# THE ARCHIVE IS RAR5
#
# Distinctions-646 ships as a .rar. Miniconda bundles bsdtar (libarchive),
# which reads RAR5, but activating an env drops base conda's bin from PATH --
# so this script looks there explicitly before giving up.
#
# Usage:
#   bash setup_d646_data.sh              # extract if needed, verify, refresh mtimes
#   bash setup_d646_data.sh --check      # verify only, change nothing
#   bash setup_d646_data.sh --force      # re-extract regardless
#
# Environment:
#   D646_ROOT     dataset root (default /scratch/mridul/data/matting/distinctions-646)
#   D646_ARCHIVE  archive path (default /scratch/mridul/data/matting/Distinctions-646.rar)

set -euo pipefail

root="${D646_ROOT:-/scratch/mridul/data/matting/distinctions-646}"
archive="${D646_ARCHIVE:-/scratch/mridul/data/matting/Distinctions-646.rar}"
mode="run"

while (( $# )); do
  case "$1" in
    --check) mode="check" ;;
    --force) mode="force" ;;
    -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

# Count FG/GT pairs in one split. Prints "<paired> <fg> <gt>".
count_pairs() {
  local dir="$root/$1"
  local fg=0 gt=0 paired=0
  if [[ -d "$dir/FG" ]]; then
    fg=$(find "$dir/FG" -maxdepth 1 -name '*.png' -type f 2>/dev/null | wc -l)
  fi
  if [[ -d "$dir/GT" ]]; then
    gt=$(find "$dir/GT" -maxdepth 1 -name '*.png' -type f 2>/dev/null | wc -l)
  fi
  if (( fg > 0 && gt > 0 )); then
    # A pair needs the same filename under both directories; comparing sorted
    # name lists catches a partial extract that counting alone would not.
    paired=$(comm -12 \
      <(find "$dir/FG" -maxdepth 1 -name '*.png' -type f -printf '%f\n' | sort) \
      <(find "$dir/GT" -maxdepth 1 -name '*.png' -type f -printf '%f\n' | sort) | wc -l)
  fi
  echo "$paired $fg $gt"
}

# Where the composites' backgrounds come from. Everything the shipped
# bg_*.txt lists name is already on this cluster: the COCO train2014 filenames
# encode the same image ids as train2017, and all 59,600 resolve there.
COCO_DIR="${D646_COCO_DIR:-/projects/ml4science/PSG_Data/coco/train2017}"
VOC_DIR="${D646_VOC_DIR:-/projects/ml4science/kazi/PartSegmentationDatasets/Pascal_VOC_2012/VOCdevkit/VOC2012/JPEGImages}"

check_backgrounds() {
  local split="$1" dir="$2" list="$root/$1/$3" ok=1
  if [[ ! -d "$dir" ]]; then
    echo "  $split backgrounds: MISSING directory $dir" >&2
    return 1
  fi
  if [[ ! -f "$list" ]]; then
    echo "  $split backgrounds: MISSING list $list" >&2
    return 1
  fi
  # The shipped lists are CRLF, so strip \r before probing.
  local total present name
  total=$(tr -d '\r' < "$list" | grep -c . || true)
  present=0
  # A file whose last line has no trailing newline loses that line to a plain
  # `read`; bg_train.txt is exactly that.
  while IFS= read -r name || [[ -n "$name" ]]; do
    name="${name%$'\r'}"
    [[ -z "$name" ]] && continue
    if [[ -f "$dir/$name" || -f "$dir/${name#COCO_train2014_}" ]]; then
      present=$((present + 1))
    fi
    # Probing 59,600 names costs a second; sampling would hide a partial set.
  done < "$list"
  if (( present < total )); then
    echo "  $split backgrounds: $present/$total found in $dir" >&2
    ok=0
  else
    echo "  $split backgrounds: $present/$total  ok"
  fi
  (( ok ))
}

find_extractor() {
  local tool
  for tool in bsdtar 7z 7za unar unrar; do
    if command -v "$tool" >/dev/null 2>&1; then
      echo "$tool"
      return 0
    fi
  done
  # Activating a conda env removes base conda's bin from PATH, and that is
  # where bsdtar lives on this cluster.
  local base
  for base in "${CONDA_EXE%/bin/conda}" /apps/common/software/Miniconda3/24.7.1-0; do
    if [[ -n "$base" && -x "$base/bin/bsdtar" ]]; then
      echo "$base/bin/bsdtar"
      return 0
    fi
  done
  return 1
}

extract_archive() {
  local tool
  if ! tool="$(find_extractor)"; then
    cat >&2 <<'EOF'
No RAR extractor found (looked for bsdtar, 7z, 7za, unar, unrar on PATH, and
for bsdtar in base conda).

Distinctions-646 ships as a RAR5 archive. Install one of these, then re-run:

    conda install -c conda-forge libarchive   # bsdtar, reads RAR5
    conda install -c conda-forge p7zip        # 7z

`pip install rarfile` alone does not work: it shells out to an unrar binary.
EOF
    return 1
  fi
  if [[ ! -f "$archive" ]]; then
    echo "Archive not found: $archive (set D646_ARCHIVE)" >&2
    return 1
  fi
  echo "  extracting with $(basename "$tool"): $(basename "$archive")"
  mkdir -p "$root"
  # Only FG, GT and the background lists are needed; the PDF and the original
  # generation scripts are not.
  case "$(basename "$tool")" in
    bsdtar)     "$tool" -x -f "$archive" -C "$root" \
                  'Distinctions-646/Train/FG/*' 'Distinctions-646/Train/GT/*' \
                  'Distinctions-646/Test/FG/*'  'Distinctions-646/Test/GT/*'  \
                  'Distinctions-646/*/*.txt' ;;
    7z|7za)     "$tool" x -y -o"$root" "$archive" >/dev/null ;;
    unar)       "$tool" -quiet -force-overwrite -output-directory "$root" "$archive" ;;
    unrar)      "$tool" x -y "$archive" "$root/" >/dev/null ;;
  esac
}

# The archive may unpack with a wrapping directory. Lift the split folders up
# to $root so the dataset's paths resolve.
normalise_layout() {
  local found
  for split in Train Test; do
    [[ -d "$root/$split" ]] && continue
    found="$(find "$root" -maxdepth 3 -type d -name "$split" 2>/dev/null | head -1)"
    if [[ -n "$found" && "$found" != "$root/$split" ]]; then
      echo "  moving $found -> $root/$split"
      mv "$found" "$root/$split"
    fi
  done
}

refresh_timestamps() {
  [[ -d "$root/$1" ]] || return 0
  find "$root/$1" -exec touch {} +
}

echo "Distinctions-646 root: $root"

if [[ "$mode" == "force" ]] || [[ ! -d "$root/Train/FG" ]] || [[ ! -d "$root/Test/FG" ]]; then
  if [[ "$mode" == "check" ]]; then
    echo "  not extracted" >&2
    echo "Distinctions-646 is NOT ready." >&2
    exit 1
  fi
  extract_archive
  normalise_layout
fi

declare -A EXPECTED=( [Train]=596 [Test]=50 )
status=0
for split in Train Test; do
  read -r paired fg gt <<<"$(count_pairs "$split")"
  if (( paired == 0 )); then
    echo "  $split: no FG/GT pairs (FG=$fg GT=$gt)" >&2
    status=1
    continue
  fi
  if (( fg != paired || gt != paired )); then
    echo "  $split: $paired pairs, but FG=$fg GT=$gt -- extract is incomplete" >&2
    status=1
    continue
  fi
  if (( paired != EXPECTED[$split] )); then
    echo "  $split: $paired pairs, expected ${EXPECTED[$split]}" >&2
    status=1
    continue
  fi
  if [[ "$mode" == "check" ]]; then
    echo "  $split: $paired FG/GT pairs  ok"
  else
    refresh_timestamps "$split"
    echo "  $split: $paired FG/GT pairs  ok (timestamps refreshed)"
  fi
done

# Composites are built at load time, so the background pools are part of the
# dataset being "ready" even though nothing here extracts them.
check_backgrounds Train "$COCO_DIR" bg_train.txt || status=1
check_backgrounds Test  "$VOC_DIR"  bg_test.txt  || status=1

if (( status != 0 )); then
  echo "Distinctions-646 is NOT ready." >&2
  exit "$status"
fi

if [[ "$mode" == "check" ]]; then
  echo "Distinctions-646 is ready (checked, nothing modified)."
else
  touch "$root"
  echo "Distinctions-646 is ready."
fi
