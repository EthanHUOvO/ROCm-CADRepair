#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
PY=${PY:-/opt/python/bin/python}
PARTS=${PARTS:-"part_001 part_002 part_003 part_004 part_005 part_006 part_007 part_008"}
TOKENS=${TOKENS:-"512 1024 2048"}
GPU=${GPU:-0}

export ROCR_VISIBLE_DEVICES="${GPU}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

for max_tokens in ${TOKENS}; do
  echo "===== Token budget ablation: ${max_tokens} ====="

  for part in ${PARTS}; do
    views=./examples/${part}/views
    out=./outputs_ablation/tokens_${max_tokens}/${part}
    mkdir -p "${out}"

    "${PY}" src/infer_one.py \
      --model "${MODEL}" \
      --views "${views}" \
      --outdir "${out}" \
      --candidate 0 \
      --max-new-tokens "${max_tokens}" \
      --do-sample \
      --temperature 0.3 \
      --profile-label "tokens_${max_tokens}"

    "${PY}" src/verify_pipeline.py \
      --code "${out}/candidate_0.py" \
      --outdir "${out}" \
      --candidate 0 || true

    if [ -f "./examples/${part}/${part}_gt.stl" ]; then
      "${PY}" src/evaluate_geometry.py \
        --gt "./examples/${part}/${part}_gt.stl" \
        --pred_dir "${out}" \
        --out "${out}/geometry_eval.csv" || true
    fi
  done
done

echo "Token ablation outputs saved under ./outputs_ablation"
