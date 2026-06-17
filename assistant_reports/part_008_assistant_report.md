# Local CAD Assistant Report: part_008

## Task Summary

- Input views: `examples/part_008/views/view_0.png` ... `view_7.png`
- Generated candidates: 5
- Successful CAD exports: 4/5
- Raw successes: 3
- Safe repair successes: 1
- Failed candidates: 1
- Best candidate: `candidate_1`
- Best STEP: `outputs/part_008/candidate_1.step`
- Best STL: `outputs/part_008/candidate_1.stl`

## Geometry Quality

- Watertight: yes
- Mean bbox relative error: 0.092
- Volume relative error: 0.159
- Normalized Chamfer distance: 0.004

## ROCm Inference Evidence

- GPU: AMD Radeon Graphics
- HIP version: 7.12.60610-2bd1678d3d
- Average inference latency: 18.04 sec
- Average generated throughput: 29.58 tokens/sec
- Peak allocated VRAM: 4.49 GB

## Candidate Status

| candidate | status | selected stage | raw error | safe error |
|---:|---|---|---|---|
| 0 | success | raw | n/a | n/a |
| 1 | success | raw | n/a | n/a |
| 2 | success | raw | n/a | n/a |
| 3 | failed | failed | OCP.OCP.StdFail.StdFail_NotDone | ValueError |
| 4 | success | safe | OCP.OCP.StdFail.StdFail_NotDone | n/a |

## Engineering Notes

- Some candidates failed in the CAD kernel. Inspect the error table before manual editing.
- Safe repair recovered at least one candidate, showing the value of execution-feedback repair.

## Error Distribution

| error type | count |
|---|---:|
| OCP.OCP.StdFail.StdFail_NotDone | 2 |
| ValueError | 1 |