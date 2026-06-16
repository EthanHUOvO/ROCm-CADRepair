# ROCm CAD Repair Competition Report

## Overall

- Samples: 8
- Candidates: 40
- Pipeline success: 34/40 (85.0%)
- Average inference latency: 19.46 sec
- Average generated throughput: n/a tokens/sec
- Average peak VRAM: 4.49 GB

## Per-Part Summary

| sample | success rate | raw | safe repair | failed | bbox error | volume error | chamfer | avg latency(s) | tokens/s | peak VRAM(GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| part_001 | 80.0% | 4 | 0 | 1 | 0.577 | 1.421 | n/a | 42.31 | n/a | 4.49 |
| part_002 | 100.0% | 4 | 1 | 0 | 0.451 | 0.932 | n/a | 18.60 | n/a | 4.49 |
| part_003 | 80.0% | 3 | 1 | 1 | 0.192 | 0.037 | n/a | 13.58 | n/a | 4.49 |
| part_004 | 100.0% | 3 | 2 | 0 | 0.498 | 0.932 | n/a | 16.44 | n/a | 4.49 |
| part_005 | 80.0% | 2 | 2 | 1 | 0.419 | 0.962 | n/a | 17.97 | n/a | 4.49 |
| part_006 | 80.0% | 3 | 1 | 1 | 0.338 | 1.518 | n/a | 12.84 | n/a | 4.49 |
| part_007 | 60.0% | 3 | 0 | 2 | 0.404 | 0.263 | n/a | 17.35 | n/a | 4.49 |
| part_008 | 100.0% | 5 | 0 | 0 | 0.201 | 0.354 | n/a | 16.57 | n/a | 4.49 |

## How To Use In The Paper

- Use latency, tokens/sec and VRAM to support the ROCm performance and resource-analysis section.
- Use raw vs safe repair counts to show the engineering contribution beyond plain model generation.
- Use bbox, volume error and Chamfer distance to discuss geometric fidelity and failure cases.
- Add rocprof and rocm-smi screenshots/logs for kernel-level evidence on the AMD test platform.
