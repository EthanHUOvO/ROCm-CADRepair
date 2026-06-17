# Local CAD Assistant Report: part_001

## Task Summary

- Input views: `examples/part_001/views/view_0.png` ... `view_7.png`
- Generated candidates: 5
- Successful CAD exports: 4/5
- Raw successes: 3
- Safe repair successes: 1
- Failed candidates: 1
- Best candidate: `candidate_3`
- Best STEP: `outputs/part_001/candidate_3.step`
- Best STL: `outputs/part_001/candidate_3.stl`

## Geometry Quality

- Watertight: yes
- Mean bbox relative error: 0.179
- Volume relative error: 0.804
- Normalized Chamfer distance: 0.021

## ROCm Inference Evidence

- GPU: AMD Radeon Graphics
- HIP version: 7.12.60610-2bd1678d3d
- Average inference latency: 14.30 sec
- Average generated throughput: 28.50 tokens/sec
- Peak allocated VRAM: 4.49 GB

## Candidate Status

| candidate | status | selected stage | raw error | safe error |
|---:|---|---|---|---|
| 0 | success | raw | n/a | n/a |
| 1 | success | safe | OCP.OCP.Standard.Standard_Failure | n/a |
| 2 | success | raw | n/a | n/a |
| 3 | success | raw | n/a | n/a |
| 4 | failed | failed | ValueError | OCP.OCP.StdFail.StdFail_NotDone |

## Engineering Notes

- Some candidates failed in the CAD kernel. Inspect the error table before manual editing.
- Safe repair recovered at least one candidate, showing the value of execution-feedback repair.

## Error Distribution

| error type | count |
|---|---:|
| OCP.OCP.Standard.Standard_Failure | 1 |
| ValueError | 1 |
| OCP.OCP.StdFail.StdFail_NotDone | 1 |