#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

LAION_META_ROOT="${LAION_META_ROOT:-$REPO_ROOT/assets/rar256_stage/laion2B-en-aesthetic}"
LAION_BATCH_ROWS="${LAION_BATCH_ROWS:-300000}"
LAION_BATCH_ID="${LAION_BATCH_ID:-0}"
LAION_WORK_ROOT="${LAION_WORK_ROOT:-$REPO_ROOT/assets/rar256_stage/laion_batches}"
LAION_IMAGE_ROOT="${LAION_IMAGE_ROOT:-$REPO_ROOT/assets/rar256_stage/laion_images_parts}"
LAION_IMAGE_SIZE="${LAION_IMAGE_SIZE:-256}"
LAION_PROCESSES="${LAION_PROCESSES:-8}"
LAION_THREADS="${LAION_THREADS:-64}"

mkdir -p "$LAION_WORK_ROOT" "$LAION_IMAGE_ROOT"

if ! python -c "import img2dataset, pyarrow" >/dev/null 2>&1; then
  python -m pip install -U img2dataset pyarrow
fi

export LAION_META_ROOT
export LAION_BATCH_ROWS
export LAION_BATCH_ID
export LAION_WORK_ROOT

python - <<'PY'
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

src = Path(os.environ["LAION_META_ROOT"])
batch_rows = int(os.environ["LAION_BATCH_ROWS"])
batch_id = int(os.environ["LAION_BATCH_ID"])
out_root = Path(os.environ["LAION_WORK_ROOT"])
out = out_root / f"laion_batch_{batch_id:04d}.parquet"

files = sorted(src.glob("*.parquet"))
if not files:
    raise SystemExit(f"No parquet files found in {src}")

columns = ["URL", "TEXT", "WIDTH", "HEIGHT", "similarity", "hash", "punsafe", "pwatermark", "aesthetic"]
start = batch_id * batch_rows
end = start + batch_rows
seen = 0
tables = []

for f in files:
    pf = pq.ParquetFile(f)
    n = pf.metadata.num_rows
    file_start = seen
    file_end = seen + n

    if file_end <= start:
        seen += n
        continue
    if file_start >= end:
        break

    table = pq.read_table(f, columns=columns)
    s = max(start - file_start, 0)
    e = min(end - file_start, n)
    tables.append(table.slice(s, e - s))
    seen += n

if not tables:
    raise SystemExit("No rows selected. Check LAION_BATCH_ID / LAION_BATCH_ROWS.")

subset = pa.concat_tables(tables)
out.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(subset, out)
print(f"wrote {subset.num_rows} rows to {out}")
PY

BATCH_NAME="$(printf "laion_batch_%04d" "$LAION_BATCH_ID")"
BATCH_PARQUET="$LAION_WORK_ROOT/${BATCH_NAME}.parquet"
BATCH_IMAGE_ROOT="$LAION_IMAGE_ROOT/batch_$(printf "%04d" "$LAION_BATCH_ID")"

img2dataset \
  --url_list "$BATCH_PARQUET" \
  --input_format parquet \
  --url_col URL \
  --caption_col TEXT \
  --output_format files \
  --output_folder "$BATCH_IMAGE_ROOT" \
  --processes_count "$LAION_PROCESSES" \
  --thread_count "$LAION_THREADS" \
  --image_size "$LAION_IMAGE_SIZE" \
  --resize_mode keep_ratio \
  --resize_only_if_bigger True \
  --skip_reencode True

du -sh "$BATCH_IMAGE_ROOT"
find "$BATCH_IMAGE_ROOT" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" -o -iname "*.bmp" \) | wc -l
