#!/usr/bin/env bash
#
# Put Distinctions-646 on disk in the layout Distinctions646MattingDataset
# expects, and keep it there.
#
#   <root>/Train_comp/merged/<foreground>.png_<k>.png
#   <root>/Train_comp/alpha/<foreground>.png_<k>.png
#   <root>/Test_comp/merged/test_<i>.png
#   <root>/Test_comp/alpha/test_<i>.png
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
# Distinctions-646 ships as a .rar, and this cluster has no extractor: no
# unrar, unar, 7z or bsdtar on PATH, none in the conda env, and no matching
# module. Install one before running this, for example:
#
#   conda install -c conda-forge libarchive   # provides bsdtar, reads RAR5
#   conda install -c conda-forge p7zip        # provides 7z
#
# `pip install rarfile` is not enough on its own -- it shells out to unrar.
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

# Count merged/alpha pairs in one split. Prints "<paired> <merged> <alpha>".
count_pairs() {
  local dir="$root/$1"
  local merged=0 alpha=0 paired=0
  if [[ -d "$dir/merged" ]]; then
    merged=$(find "$dir/merged" -maxdepth 1 -name '*.png' -type f 2>/dev/null | wc -l)
  fi
  if [[ -d "$dir/alpha" ]]; then
    alpha=$(find "$dir/alpha" -maxdepth 1 -name '*.png' -type f 2>/dev/null | wc -l)
  fi
  if (( merged > 0 && alpha > 0 )); then
    # A pair needs the same filename under both directories; comparing sorted
    # name lists catches a partial extract that counting alone would not.
    paired=$(comm -12 \
      <(find "$dir/merged" -maxdepth 1 -name '*.png' -type f -printf '%f\n' | sort) \
      <(find "$dir/alpha"  -maxdepth 1 -name '*.png' -type f -printf '%f\n' | sort) | wc -l)
  fi
  echo "$paired $merged $alpha"
}

find_extractor() {
  local tool
  for tool in bsdtar 7z 7za unar unrar; do
    if command -v "$tool" >/dev/null 2>&1; then
      echo "$tool"
      return 0
    fi
  done
  return 1
}

extract_archive() {
  local tool
  if ! tool="$(find_extractor)"; then
    cat >&2 <<'EOF'
No RAR extractor found (looked for bsdtar, 7z, 7za, unar, unrar).

Distinctions-646 ships as a RAR5 archive. Install one of these into the active
environment, then re-run this script:

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
  echo "  extracting with $tool: $(basename "$archive")"
  mkdir -p "$root"
  case "$tool" in
    bsdtar)     bsdtar -x -f "$archive" -C "$root" ;;
    7z|7za)     "$tool" x -y -o"$root" "$archive" >/dev/null ;;
    unar)       unar -quiet -force-overwrite -output-directory "$root" "$archive" ;;
    unrar)      unrar x -y "$archive" "$root/" >/dev/null ;;
  esac
}

# The archive may unpack with a wrapping directory. Lift the split folders up
# to $root so the dataset's paths resolve.
normalise_layout() {
  local found
  for split in Train_comp Test_comp; do
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

if [[ "$mode" == "force" ]] || [[ ! -d "$root/Train_comp/merged" ]] || [[ ! -d "$root/Test_comp/merged" ]]; then
  if [[ "$mode" == "check" ]]; then
    echo "  not extracted" >&2
    echo "Distinctions-646 is NOT ready." >&2
    exit 1
  fi
  extract_archive
  normalise_layout
fi

status=0
for split in Train_comp Test_comp; do
  read -r paired merged alpha <<<"$(count_pairs "$split")"
  if (( paired == 0 )); then
    echo "  $split: no pairs (merged=$merged alpha=$alpha)" >&2
    status=1
    continue
  fi
  if (( merged != paired || alpha != paired )); then
    echo "  $split: $paired pairs, but merged=$merged alpha=$alpha -- extract is incomplete" >&2
    status=1
    continue
  fi
  if [[ "$mode" == "check" ]]; then
    echo "  $split: $paired pairs  ok"
  else
    refresh_timestamps "$split"
    echo "  $split: $paired pairs  ok (timestamps refreshed)"
  fi
done

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
