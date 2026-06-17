# Local CAD Assistant Report: part_005

## Task Summary

- Input views: `examples/part_005/views/view_0.png` ... `view_7.png`
- Generated candidates: 5
- Successful CAD exports: 5/5
- Raw successes: 4
- Safe repair successes: 1
- Failed candidates: 0
- Best candidate: `candidate_1`
- Best STEP: `outputs/part_005/candidate_1.step`
- Best STL: `outputs/part_005/candidate_1.stl`

## Geometry Quality

- Watertight: yes
- Mean bbox relative error: 0.359
- Volume relative error: 0.007
- Normalized Chamfer distance: 0.016

## ROCm Inference Evidence

- GPU: AMD Radeon Graphics
- HIP version: 7.12.60610-2bd1678d3d
- Average inference latency: 18.14 sec
- Average generated throughput: 29.81 tokens/sec
- Peak allocated VRAM: 4.49 GB

## Candidate Status

| candidate | status | selected stage | raw error | safe error |
|---:|---|---|---|---|
| 0 | success | raw | n/a | n/a |
| 1 | success | raw | n/a | n/a |
| 2 | success | raw | n/a | n/a |
| 3 | success | raw | n/a | n/a |
| 4 | success | safe | OCP.OCP.StdFail.StdFail_NotDone | n/a |

## Engineering Notes

- All generated candidates passed export validation.
- Safe repair recovered at least one candidate, showing the value of execution-feedback repair.

## Error Distribution

| error type | count |
|---|---:|
| ValueError | 3 |
| OCP.OCP.StdFail.StdFail_NotDone | 1 |