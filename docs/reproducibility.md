# Reproducibility Notes

## Target Platform

The competition topic 2 reference environment is expected to provide:

- AMD Radeon PRO W series GPUs
- ROCm 7.1.1 or the version announced by the organizer
- PyTorch built for ROCm
- Python with CadQuery and scientific Python packages

Always record the exact environment before reporting results:

```bash
/opt/python/bin/python src/check_rocm.py
rocminfo | head -n 80
rocm-smi --showproductname --showdriverversion --showmeminfo vram
```

## Python Dependencies

Install project-level Python packages:

```bash
pip install -r requirements.txt
```

Install PyTorch according to the ROCm version on the AMD machine. The PyTorch
wheel must match the ROCm runtime provided by the competition environment.

## Model

Download the Zero-to-CAD model to:

```text
models/Zero-To-CAD-Qwen3-VL-2B
```

or pass another path through:

```bash
MODEL=/path/to/model bash scripts/run_8parts_benchmark.sh
```

## Minimal Reproduction

```bash
/opt/python/bin/python src/check_rocm.py
PART=part_001 CANDIDATE=0 GPU=0 bash scripts/run_rocm_profile.sh
/opt/python/bin/python src/build_competition_report.py
```

Expected artifacts:

- `outputs/part_001/infer_candidate_0.json`
- `outputs/part_001/pipeline_candidate_0.json`
- `profiles/part_001_candidate0_gpu0/*`
- `docs/results/competition_report.md`

## Full Reproduction

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
NUM_CANDIDATES=5 \
MAX_TOKENS=1024 \
bash scripts/run_8parts_benchmark.sh

/opt/python/bin/python src/collect_inference_stats.py
/opt/python/bin/python src/collect_overall_results.py
/opt/python/bin/python src/analyze_success_gain.py
/opt/python/bin/python src/analyze_geometry_quality.py
/opt/python/bin/python src/build_competition_report.py
```

## What To Submit

- Code repository
- `docs/results/competition_report.md`
- `docs/results/*.csv`
- `profiles/*` ROCm profiling artifacts
- Selected generated STEP/STL examples
- PPT/video screenshots showing ROCm profiler, terminal logs, and CAD outputs
