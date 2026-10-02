#!/usr/bin/env bash
# Rebuild every table and figure in docs/ from scratch (needs KITTI labels under data/).
# Usage: scripts/reproduce.sh [N_SEEDS]   (default 3)
set -euo pipefail
SEEDS=${1:-3}
CACHE=outputs/cache
mkdir -p outputs

for h in 10 30; do
  specs=()
  for variant in "Transformer:transformer:" "Social:social:" \
                 "Transformer + ridge prior:transformer:--ridge-prior" \
                 "Social + ridge prior:social:--ridge-prior"; do
    IFS=: read -r name arch flag <<<"$variant"
    tag=$(echo "${name}" | tr -c 'a-zA-Z0-9\n' '_')
    paths=()
    for s in $(seq 0 $((SEEDS - 1))); do
      out="outputs/${tag}_h${h}_s${s}.pt"
      # shellcheck disable=SC2086
      python -m cftraj train --arch "$arch" --future-len "$h" $flag --seed "$s" \
        --cache-dir "$CACHE" --out "$out"
      paths+=("$out")
    done
    specs+=(--ckpt "${name}=$(IFS=,; echo "${paths[*]}")")
  done
  python -m cftraj evaluate --cache-dir "$CACHE" "${specs[@]}"
done

python -m cftraj crossval --folds 5 --out docs/crossval.md --cache-dir "$CACHE"
python -m cftraj attention --labels data/training/label_02/0019.txt
echo "Done: see docs/"
