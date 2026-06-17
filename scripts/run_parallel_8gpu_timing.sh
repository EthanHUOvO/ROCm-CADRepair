#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
PY=${PY:-/opt/python/bin/python}
MAX_TOKENS=${MAX_TOKENS:-512}
CANDIDATE=${CANDIDATE:-0}
OUT_ROOT=${OUT_ROOT:-./outputs_timing/parallel_8gpu}

PARTS=(part_001 part_002 part_003 part_004 part_005 part_006 part_007 part_008)
mkdir -p "${OUT_ROOT}" logs docs/results

START=$(date +%s)

for gpu in 0 1 2 3 4 5 6 7; do
  part=${PARTS[$gpu]}
  out="${OUT_ROOT}/${part}"
  mkdir -p "${out}"

  (
    export ROCR_VISIBLE_DEVICES="${gpu}"
    unset HIP_VISIBLE_DEVICES
    unset CUDA_VISIBLE_DEVICES

    echo "===== Parallel run: ${part} on physical GPU ${gpu} ====="

    "${PY}" src/infer_one.py \
      --model "${MODEL}" \
      --views "./examples/${part}/views" \
      --outdir "${out}" \
      --candidate "${CANDIDATE}" \
      --max-new-tokens "${MAX_TOKENS}" \
      --do-sample \
      --temperature 0.3 \
      --profile-label "parallel_8gpu_gpu${gpu}"

    "${PY}" src/verify_pipeline.py \
      --code "${out}/candidate_${CANDIDATE}.py" \
      --outdir "${out}" \
      --candidate "${CANDIDATE}" || true
  ) > "logs/${part}_parallel_gpu${gpu}.log" 2>&1 &
done

wait

END=$(date +%s)
WALL=$((END - START))

cat > docs/results/parallel_8gpu_timing.csv <<CSV
mode,gpus,parts,candidate,max_tokens,wall_time_sec
parallel,8,8,${CANDIDATE},${MAX_TOKENS},${WALL}
CSV

if [ -f docs/results/serial_8parts_timing.csv ]; then
  "${PY}" src/compare_gpu_timing.py \
    --serial docs/results/serial_8parts_timing.csv \
    --parallel docs/results/parallel_8gpu_timing.csv \
    --out docs/results/gpu_timing_comparison.csv
fi

echo "Parallel wall time: ${WALL} sec"
echo "Saved: docs/results/parallel_8gpu_timing.csv"
