#!/usr/bin/env bash
#
# Put AM-2K on disk in the layout AM2KMattingDataset expects, and keep it there.
#
#   <root>/am2k_split_category.json
#   <root>/train/original/<sample_id>.jpg        1800 pairs
#   <root>/train/mask/<sample_id>.png
#   <root>/validation/original/<sample_id>.jpg    200 pairs
#   <root>/validation/mask/<sample_id>.png
#
# Idempotent: run it before every session. When the data is already complete it
# only refreshes timestamps and exits in a second or two.
#
# WHY THE DATA KEEPS DISAPPEARING
#
# The AM-2K archives were built in 2021, and `unzip` preserves archive
# timestamps, so a plain extract stamps every image with mtime 2021-07-01 --
# five years old on arrival. Any scratch cleanup that reaps files by age
# deletes them on its next sweep, while the .zip files (current mtimes) survive
# untouched. That is exactly the pattern we kept seeing: images gone, zips
# fine. So this script touches every extracted file and directory afterwards.
# The touch is the point, not a nicety -- do not drop it.
#
# If it still vanishes, the copy is being reaped for some other reason and the
# fix is to move the dataset off scratch entirely (see AM2K_ROOT below).
#
# Usage:
#   bash setup_am2k_data.sh              # extract if needed, verify, refresh mtimes
#   bash setup_am2k_data.sh --check      # verify only, change nothing
#   bash setup_am2k_data.sh --force      # re-extract even if complete
#
# Environment:
#   AM2K_ROOT  dataset root (default /scratch/mridul/data/matting/am-2k).
#              Point this at /projects/<alloc>/... for a copy that is not
#              subject to scratch cleanup, and set it in the training env too.

set -euo pipefail

root="${AM2K_ROOT:-/scratch/mridul/data/matting/am-2k}"
metadata="$root/am2k_split_category.json"
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

python_bin="${CONDA_PREFIX:+$CONDA_PREFIX/bin/python}"
[[ -x "${python_bin:-}" ]] || python_bin="$(command -v python3)"

if [[ ! -d "$root" ]]; then
  echo "AM-2K root does not exist: $root" >&2
  echo "Set AM2K_ROOT, or restore the directory holding the .zip archives." >&2
  exit 1
fi
if [[ ! -f "$metadata" ]]; then
  echo "Missing $metadata -- the dataset cannot be verified without it." >&2
  exit 1
fi

# Reports every sample id in the split metadata that has no image or no alpha
# on disk. Silence means the split is complete.
verify_split() {
  "$python_bin" - "$root" "$1" <<'PY'
import json, os, sys

root, split = sys.argv[1], sys.argv[2]
with open(os.path.join(root, "am2k_split_category.json"), encoding="utf-8") as handle:
    metadata = json.load(handle)

expected = sorted(k for k, v in metadata.items() if v.get("split") == split)
missing = [
    sample_id
    for sample_id in expected
    if not os.path.isfile(os.path.join(root, split, "original", f"{sample_id}.jpg"))
    or not os.path.isfile(os.path.join(root, split, "mask", f"{sample_id}.png"))
]
print(f"{len(expected) - len(missing)} {len(expected)}", file=sys.stderr)
for sample_id in missing[:5]:
    print(sample_id)
if len(missing) > 5:
    print(f"...and {len(missing) - 5} more")
PY
}

# Two values per split: how many pairs are present, and how many are expected.
split_counts() {
  verify_split "$1" 2>&1 >/dev/null
}

split_missing() {
  verify_split "$1" 2>/dev/null
}

extract_split() {
  local split="$1" found=0 archive
  shopt -s nullglob
  # One glob covers both the single-file and the split-download naming
  # (train.zip as well as train-<timestamp>-1-001.zip).
  for archive in "$root/$split"*.zip; do
    [[ -f "$archive" ]] || continue
    found=1
    echo "  unzip $(basename "$archive")"
    # Only original/ and mask/ are used by AM2KMattingDataset. Skipping bg/,
    # fg/ and trimap/ saves roughly half the bytes and half the time.
    #
    # unzip exits 11 when an archive holds none of the requested members, which
    # is normal for a multi-part download, so it is not an error here.
    set +e
    unzip -o -q "$archive" "$split/original/*" "$split/mask/*" -d "$root"
    local status=$?
    set -e
    if (( status != 0 && status != 11 )); then
      echo "unzip failed on $archive (exit $status)" >&2
      return 1
    fi
  done
  shopt -u nullglob
  if (( ! found )); then
    echo "No archives matching $root/$split-*.zip -- cannot rebuild $split." >&2
    return 1
  fi
}

# Extracted files inherit the archive's 2021 timestamps; a scratch reaper that
# works on age will delete them almost immediately. Make them look new.
refresh_timestamps() {
  local split="$1"
  [[ -d "$root/$split" ]] || return 0
  find "$root/$split" -exec touch {} +
}

echo "AM-2K root: $root"
status=0
for split in train validation; do
  counts="$(split_counts "$split")"
  present="${counts%% *}"
  expected="${counts##* }"

  if [[ "$mode" == "force" ]] || (( present != expected )); then
    if [[ "$mode" == "check" ]]; then
      echo "  $split: $present/$expected pairs -- INCOMPLETE"
      split_missing "$split" | sed 's/^/    missing: /'
      status=1
      continue
    fi
    if [[ "$mode" == "force" ]]; then
      echo "  $split: $present/$expected pairs, re-extracting (--force)"
    else
      echo "  $split: $present/$expected pairs, extracting"
    fi
    extract_split "$split"
    counts="$(split_counts "$split")"
    present="${counts%% *}"
    expected="${counts##* }"
    if (( present != expected )); then
      echo "  $split: still $present/$expected pairs after extraction" >&2
      split_missing "$split" | sed 's/^/    missing: /' >&2
      status=1
      continue
    fi
  fi

  if [[ "$mode" == "check" ]]; then
    echo "  $split: $present/$expected pairs  ok"
  else
    refresh_timestamps "$split"
    echo "  $split: $present/$expected pairs  ok (timestamps refreshed)"
  fi
done

if (( status != 0 )); then
  echo "AM-2K is NOT ready." >&2
  exit "$status"
fi

if [[ "$mode" == "check" ]]; then
  echo "AM-2K is ready (checked, nothing modified)."
else
  touch "$metadata" "$root"
  echo "AM-2K is ready."
fi
