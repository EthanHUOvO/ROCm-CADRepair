#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
PY=${PY:-/opt/python/bin/python}
MAX_TOKENS=${MAX_TOKENS:-512}
CANDIDATE=${CANDIDATE:-0}
GPU=${GPU:-0}
OUT_ROOT=${OUT_ROOT:-./outputs_timing/serial}

PARTS=(part_001 part_002 part_003 part_004 part_005 part_006 part_007 part_008)
mkdir -p "${OUT_ROOT}" docs/results

export ROCR_VISIBLE_DEVICES="${GPU}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

START=$(date +%s)

for part in "${PARTS[@]}"; do
  OUT="${OUT_ROOT}/${part}"
  mkdir -p "${OUT}"
  echo "===== Serial run: ${part} on GPU ${GPU} ====="

  "${PY}" src/infer_one.py \
    --model "${MODEL}" \
    --views "./examples/${part}/views" \
    --outdir "${OUT}" \
    --candidate "${CANDIDATE}" \
    --max-new-tokens "${MAX_TOKENS}" \
    --do-sample \
    --temperature 0.3 \
    --profile-label "serial_8parts_gpu${GPU}"

  "${PY}" src/verify_pipeline.py \
    --code "${OUT}/candidate_${CANDIDATE}.py" \
    --outdir "${OUT}" \
    --candidate "${CANDIDATE}" || true
done

END=$(date +%s)
WALL=$((END - START))

cat > docs/results/serial_8parts_timing.csv <<CSV
mode,gpus,parts,candidate,max_tokens,wall_time_sec
serial,1,8,${CANDIDATE},${MAX_TOKENS},${WALL}
CSV

echo "Serial wall time: ${WALL} sec"
echo "Saved: docs/results/serial_8parts_timing.csv"
