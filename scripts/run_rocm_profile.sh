#!/bin/bash
set -euo pipefail

MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
PY=${PY:-/opt/python/bin/python}
PART=${PART:-part_001}
CANDIDATE=${CANDIDATE:-0}
MAX_TOKENS=${MAX_TOKENS:-1024}
TEMPERATURE=${TEMPERATURE:-0.3}
GPU=${GPU:-0}

VIEWS=./examples/${PART}/views
OUT=./outputs/${PART}
PROFILE_DIR=./profiles/${PART}_candidate${CANDIDATE}_gpu${GPU}

mkdir -p "${OUT}" "${PROFILE_DIR}"

if [ ! -d "${VIEWS}" ]; then
  echo "Missing views folder: ${VIEWS}"
  exit 1
fi

export ROCR_VISIBLE_DEVICES="${GPU}"
unset HIP_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

echo "===== ROCm CAD Repair Profile ====="
echo "PART=${PART}"
echo "CANDIDATE=${CANDIDATE}"
echo "GPU=${GPU}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "PROFILE_DIR=${PROFILE_DIR}"

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname --showdriverversion --showmeminfo vram > "${PROFILE_DIR}/rocm_smi_before.txt" || true
fi

RUN_CMD=(
  "${PY}" src/infer_one.py
  --model "${MODEL}"
  --views "${VIEWS}"
  --outdir "${OUT}"
  --candidate "${CANDIDATE}"
  --max-new-tokens "${MAX_TOKENS}"
  --do-sample
  --temperature "${TEMPERATURE}"
  --profile-label "rocm_profile_${PART}_gpu${GPU}"
)

if command -v rocprofv3 >/dev/null 2>&1; then
  echo "Using rocprofv3"
  rocprofv3 --output-directory "${PROFILE_DIR}" -- "${RUN_CMD[@]}"
elif command -v rocprof >/dev/null 2>&1; then
  echo "Using rocprof"
  rocprof --stats -o "${PROFILE_DIR}/rocprof_stats.csv" "${RUN_CMD[@]}"
else
  echo "rocprof/rocprofv3 not found; running inference without kernel profiler."
  "${RUN_CMD[@]}"
fi

if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showuse --showmemuse --showtemp --showpower > "${PROFILE_DIR}/rocm_smi_after.txt" || true
fi

"${PY}" src/verify_pipeline.py \
  --code "${OUT}/candidate_${CANDIDATE}.py" \
  --outdir "${OUT}" \
  --candidate "${CANDIDATE}" || true

echo "Profile artifacts saved to ${PROFILE_DIR}"
