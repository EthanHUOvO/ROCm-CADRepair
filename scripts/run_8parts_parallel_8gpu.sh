#!/bin/bash
set -e

MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
PY=${PY:-/opt/python/bin/python}
MAX_TOKENS=${MAX_TOKENS:-512}
CANDIDATE=${CANDIDATE:-0}

PARTS=(part_001 part_002 part_003 part_004 part_005 part_006 part_007 part_008)

mkdir -p logs

echo "Run 8 parts in parallel on 8 GPUs"
echo "MODEL=${MODEL}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "CANDIDATE=${CANDIDATE}"

for gpu in 0 1 2 3 4 5 6 7
do
  part=${PARTS[$gpu]}
  views=./examples/${part}/views
  out=./outputs/${part}

  mkdir -p ${out}

  echo "Launch ${part} on physical GPU ${gpu}"

  (
    unset HIP_VISIBLE_DEVICES
    unset CUDA_VISIBLE_DEVICES
    export ROCR_VISIBLE_DEVICES=${gpu}

    echo "===== Runtime GPU check for ${part} on physical GPU ${gpu} ====="
    ${PY} - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("available:", torch.cuda.is_available())
print("count:", torch.cuda.device_count())
if torch.cuda.device_count() > 0:
    print("visible gpu name:", torch.cuda.get_device_name(0))
PY

    ${PY} src/infer_one.py \
      --model ${MODEL} \
      --views ${views} \
      --outdir ${out} \
      --candidate ${CANDIDATE} \
      --max-new-tokens ${MAX_TOKENS} \
      --do-sample \
      --temperature 0.3

    ${PY} src/verify_pipeline.py \
      --code ${out}/candidate_${CANDIDATE}.py \
      --outdir ${out} \
      --candidate ${CANDIDATE}
  ) > logs/${part}_gpu${gpu}_candidate${CANDIDATE}.log 2>&1 &

done

wait

echo "All 8 GPU jobs finished."

for part in "${PARTS[@]}"
do
  echo "===== ${part} ====="
  if [ -f outputs/${part}/pipeline_candidate_${CANDIDATE}.json ]; then
    cat outputs/${part}/pipeline_candidate_${CANDIDATE}.json
  else
    echo "No pipeline result found."
  fi
done
