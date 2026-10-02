#!/usr/bin/env bash
# Rebuild every table and figure in docs/ from scratch (needs KITTI labels under data/).
# Usage: scripts/reproduce.sh [N_SEEDS]   (default 3; ~1 h on a laptop CPU)
set -euo pipefail
SEEDS=${1:-3}
CACHE=outputs/cache
mkdir -p outputs

for h in 10 30; do
  arch=$([ "$h" = 10 ] && echo transformer || echo social)
  ckpts=()
  for s in $(seq 0 $((SEEDS - 1))); do
    python -m cftraj train --arch "$arch" --future-len "$h" --ridge-prior --seed "$s" \
      --cache-dir "$CACHE" --out "outputs/${arch}_h${h}_s${s}.pt"
    ckpts+=("outputs/${arch}_h${h}_s${s}.pt")
  done
  python -m cftraj evaluate --cache-dir "$CACHE" \
    --ckpt "${arch}+ridge=$(IFS=,; echo "${ckpts[*]}")"
done

python -m cftraj crossval --folds 5 --out docs/crossval.md --cache-dir "$CACHE"
python -m cftraj attention --labels data/training/label_02/0019.txt
echo "Done: see docs/"
