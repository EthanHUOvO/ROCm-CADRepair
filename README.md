# ROCm-CADRepair / zero2CAD

This project turns Autodesk Zero-to-CAD into a ROCm-oriented CAD repair and
validation pipeline for the AMD ROCm competition topic 2.

The core idea is not only to generate CadQuery code from multi-view CAD images,
but also to verify whether the generated CAD is executable, exportable, and close
to the target geometry. This makes the project suitable for a research workflow:
ROCm accelerates multimodal generation, while the downstream CAD pipeline
quantifies success rate, repair gain, geometric fidelity, latency, throughput,
and VRAM usage.

## Pipeline

1. Multi-view CAD images are passed into Zero-to-CAD/Qwen-VL on an AMD GPU.
2. The model generates CadQuery Python code.
3. The raw code is executed and exported to STEP/STL.
4. Failed code is retried through a safer CadQuery execution path.
5. STL outputs are compared with ground truth geometry.
6. Results are summarized for paper/PPT use.

## Competition Value

- ROCm usage: PyTorch ROCm runs the multimodal model on Radeon/ROCm GPUs.
- CAD research workflow: generation is followed by executable validation and
  geometric quality measurement.
- Engineering contribution: the safe repair stage improves usable CAD output
  beyond plain model sampling.
- Performance evidence: logs include latency, generated tokens/sec, peak VRAM,
  PyTorch version, HIP version, GPU name, and visible ROCm devices.
- ROCm post-processing: optional Chamfer distance uses PyTorch tensor distance
  computation on the AMD GPU for geometry-quality analysis.
- Reproducibility: scripts run the benchmark, token ablation, 8-GPU parallel
  inference, and ROCm profiling.

## Quick Start

Check ROCm/PyTorch:

```bash
/opt/python/bin/python src/check_rocm.py
```

Run the 8-part benchmark:

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
NUM_CANDIDATES=5 \
MAX_TOKENS=1024 \
bash scripts/run_8parts_benchmark.sh
```

Run the benchmark with ROCm Chamfer geometry evaluation:

```bash
MODEL=./models/Zero-To-CAD-Qwen3-VL-2B \
NUM_CANDIDATES=5 \
MAX_TOKENS=1024 \
ENABLE_CHAMFER=1 \
bash scripts/run_8parts_benchmark.sh
```

Collect aggregate results:

```bash
/opt/python/bin/python src/collect_inference_stats.py
/opt/python/bin/python src/collect_overall_results.py
/opt/python/bin/python src/analyze_success_gain.py
/opt/python/bin/python src/analyze_geometry_quality.py
/opt/python/bin/python src/build_competition_report.py
```

Run one ROCm profile:

```bash
PART=part_001 CANDIDATE=0 GPU=0 MAX_TOKENS=1024 \
bash scripts/run_rocm_profile.sh
```

Run token-budget ablation:

```bash
GPU=0 TOKENS="512 1024 2048" bash scripts/run_token_ablation.sh
```

## Important Outputs

- `outputs/part_*/candidate_*.py`: generated CadQuery programs
- `outputs/part_*/candidate_*.step`: generated STEP files
- `outputs/part_*/candidate_*.stl`: generated STL files
- `outputs/part_*/infer_candidate_*.json`: ROCm inference metrics
- `outputs/part_*/pipeline_summary.csv`: raw/safe/fail status
- `outputs/part_*/geometry_eval.csv`: bbox, volume, watertight metrics
- `outputs/part_*/chamfer_eval.csv`: ROCm/PyTorch Chamfer distance metrics
- `docs/results/competition_report.md`: paper-ready summary
- `profiles/*`: rocprof and rocm-smi artifacts

## Suggested Paper Positioning

Recommended title:

> ROCm-Accelerated Zero-to-CAD Repair and Verification for Scientific CAD
> Reconstruction

Recommended claim:

> The system extends Zero-to-CAD from a visual-to-code generator into a
> reproducible ROCm CAD reconstruction workflow with executable verification,
> geometric evaluation, profiling, and ablation analysis.
