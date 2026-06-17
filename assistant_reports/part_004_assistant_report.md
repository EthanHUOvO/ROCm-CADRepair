# Local CAD Assistant Report: part_004

## Task Summary

- Input views: `examples/part_004/views/view_0.png` ... `view_7.png`
- Generated candidates: 5
- Successful CAD exports: 5/5
- Raw successes: 4
- Safe repair successes: 1
- Failed candidates: 0
- Best candidate: `candidate_2`
- Best STEP: `outputs/part_004/candidate_2.step`
- Best STL: `outputs/part_004/candidate_2.stl`

## Geometry Quality

- Watertight: yes
- Mean bbox relative error: 0.395
- Volume relative error: 2.612
- Normalized Chamfer distance: 0.033

## ROCm Inference Evidence

- GPU: AMD Radeon Graphics
- HIP version: 7.12.60610-2bd1678d3d
- Average inference latency: 15.84 sec
- Average generated throughput: 28.91 tokens/sec
- Peak allocated VRAM: 4.49 GB

## Candidate Status

| candidate | status | selected stage | raw error | safe error |
|---:|---|---|---|---|
| 0 | success | safe | OCP.OCP.Standard.Standard_Failure | n/a |
| 1 | success | raw | n/a | n/a |
| 2 | success | raw | n/a | n/a |
| 3 | success | raw | n/a | n/a |
| 4 | success | raw | n/a | n/a |

## Engineering Notes

- All generated candidates passed export validation.
- The best candidate has a large volume error; check missing holes, bosses, and support features.
- Safe repair recovered at least one candidate, showing the value of execution-feedback repair.

## Error Distribution

| error type | count |
|---|---:|
| OCP.OCP.Standard.Standard_Failure | 1 |