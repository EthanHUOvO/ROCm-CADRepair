#!/bin/bash
set -e

MODEL=${MODEL:-./models/Zero-To-CAD-Qwen3-VL-2B}
NUM_CANDIDATES=${NUM_CANDIDATES:-5}
MAX_TOKENS=${MAX_TOKENS:-1024}
PY=${PY:-/opt/python/bin/python}
ENABLE_CHAMFER=${ENABLE_CHAMFER:-0}
CHAMFER_POINTS=${CHAMFER_POINTS:-4096}

for p in part_001 part_002 part_003 part_004 part_005 part_006 part_007 part_008
do
  echo "=============================="
  echo "Running benchmark for ${p}"
  echo "=============================="

  VIEWS=./examples/${p}/views
  OUT=./outputs/${p}
  mkdir -p ${OUT}

  if [ ! -d "${VIEWS}" ]; then
    echo "Missing views folder: ${VIEWS}"
    echo "Please run: /opt/python/bin/python src/create_benchmark_parts.py"
    exit 1
  fi

  for i in $(seq 0 $((NUM_CANDIDATES-1)))
  do
    TEMP=$(python3 - <<PY
print(round(0.2 + 0.1 * ${i}, 2))
PY
)

    echo "Generate ${p}, candidate ${i}, temperature=${TEMP}"

    ${PY} src/infer_one.py \
      --model ${MODEL} \
      --views ${VIEWS} \
      --outdir ${OUT} \
      --candidate ${i} \
      --max-new-tokens ${MAX_TOKENS} \
      --do-sample \
      --temperature ${TEMP}

    ${PY} src/verify_pipeline.py \
      --code ${OUT}/candidate_${i}.py \
      --outdir ${OUT} \
      --candidate ${i} || true
  done

  ${PY} src/summarize_pipeline.py \
    --outdir ${OUT} \
    --csv ${OUT}/pipeline_summary.csv || true

  ${PY} src/error_analysis.py \
    --outdir ${OUT} \
    --csv ${OUT}/error_analysis.csv || true

  if [ -f ./examples/${p}/${p}_gt.stl ]; then
    ${PY} src/evaluate_geometry.py \
      --gt ./examples/${p}/${p}_gt.stl \
      --pred_dir ${OUT} \
      --out ${OUT}/geometry_eval.csv || true

    if [ -f ${OUT}/geometry_eval.csv ]; then
      ${PY} src/select_best_candidate.py \
        --geometry_csv ${OUT}/geometry_eval.csv \
        --out ${OUT}/best_candidate.json || true
    fi

    if [ "${ENABLE_CHAMFER}" = "1" ]; then
      ${PY} src/evaluate_chamfer_rocm.py \
        --gt ./examples/${p}/${p}_gt.stl \
        --pred-dir ${OUT} \
        --out ${OUT}/chamfer_eval.csv \
        --points ${CHAMFER_POINTS} || true
    fi
  fi
done
