#!/bin/bash
# Batch: for every page, analyze (real OCR + candidates) -> plan (VLM) -> reconstruct_elements.
# Resumable: a page whose calibrated SlideIR already exists is skipped.
set -u
SRC=/data/pilot/deck-src
OUT=/data/pilot/deck
STORE=/data/pilot/store2
mkdir -p "$OUT" "$STORE"
CLI='import sys;from image2ppt_vision.cli import main;sys.exit(main())'

PAGES=$(cd "$SRC" && ls -1 *.png | sort)
echo "[batch] pages: $(echo $PAGES | wc -w) | model=${PLANNER_MODEL:-?} | $(date -Is)"

for f in $PAGES; do
  id="${f%%_*}"
  dir="$OUT/$id"
  mkdir -p "$dir"
  if [ -f "$dir/recon/calibrated/slideir.json" ]; then
    echo "[skip] $id already done"
    continue
  fi
  echo "=== [$id] $(date -Is) analyze"
  if ! python3 -c "$CLI" --data "$STORE" analyze \
        --input "$SRC/$f" --options /data/pilot/opts.json > "$dir/analyze.json" 2> "$dir/analyze.err"; then
    echo "[fail] $id analyze"; tail -3 "$dir/analyze.err"; continue
  fi
  JOB=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['id'])" "$dir/analyze.json")
  INV="$STORE/jobs/$JOB/elements/inventory.json"
  if [ ! -f "$INV" ]; then echo "[fail] $id no inventory"; continue; fi
  echo "[$id] job=$JOB analyze ok"

  echo "=== [$id] plan"
  if ! python3 /opt/src/plan_only.py "$INV" "$SRC/$f" "$dir/plan" > "$dir/plan.out" 2>&1; then
    echo "[fail] $id plan"; tail -4 "$dir/plan.out"; continue
  fi
  echo "[$id] plan ok: $(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print('attempts',d['attempts'],'nodes',d['objects'],'texts',d['texts_total'],'secs',d['elapsed_s'])" "$dir/plan.out" 2>/dev/null)"

  echo "=== [$id] reconstruct"
  rm -rf "$dir/recon"
  if ! python3 -c "$CLI" --data "$STORE" reconstruct_elements \
        --input "$SRC/$f" --plan "$dir/plan/plan.json" --analysis-job-id "$JOB" \
        --output "$dir/recon" --renderer-version 2.8.129 > "$dir/recon.json" 2> "$dir/recon.err"; then
    echo "[fail] $id reconstruct"; tail -4 "$dir/recon.err"; continue
  fi
  echo "[$id] reconstruct ok"
done

echo "[batch] done $(date -Is)"
