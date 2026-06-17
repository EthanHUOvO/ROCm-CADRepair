# Local CAD Assistant Report: part_002

## Task Summary

- Input views: `examples/part_002/views/view_0.png` ... `view_7.png`
- Generated candidates: 5
- Successful CAD exports: 4/5
- Raw successes: 3
- Safe repair successes: 1
- Failed candidates: 1
- Best candidate: `candidate_1`
- Best STEP: `outputs/part_002/candidate_1.step`
- Best STL: `outputs/part_002/candidate_1.stl`

## Geometry Quality

- Watertight: yes
- Mean bbox relative error: 0.396
- Volume relative error: 0.030
- Normalized Chamfer distance: 0.017

## ROCm Inference Evidence

- GPU: AMD Radeon Graphics
- HIP version: 7.12.60610-2bd1678d3d
- Average inference latency: 13.59 sec
- Average generated throughput: 28.53 tokens/sec
- Peak allocated VRAM: 4.49 GB

## Candidate Status

| candidate | status | selected stage | raw error | safe error |
|---:|---|---|---|---|
| 0 | success | safe | OCP.OCP.Standard.Standard_Failure | n/a |
| 1 | success | raw | n/a | n/a |
| 2 | failed | failed | IndexError | IndexError |
| 3 | success | raw | n/a | n/a |
| 4 | success | raw | n/a | n/a |

## Engineering Notes

- Some candidates failed in the CAD kernel. Inspect the error table before manual editing.
- Safe repair recovered at least one candidate, showing the value of execution-feedback repair.

## Error Distribution

| error type | count |
|---|---:|
| IndexError | 2 |
| OCP.OCP.Standard.Standard_Failure | 1 |
| NameError | 1 |
| ValueError | 1 |