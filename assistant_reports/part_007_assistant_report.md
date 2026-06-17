# Local CAD Assistant Report: part_007

## Task Summary

- Input views: `examples/part_007/views/view_0.png` ... `view_7.png`
- Generated candidates: 5
- Successful CAD exports: 2/5
- Raw successes: 2
- Safe repair successes: 0
- Failed candidates: 3
- Best candidate: `candidate_1`
- Best STEP: `outputs/part_007/candidate_1.step`
- Best STL: `outputs/part_007/candidate_1.stl`

## Geometry Quality

- Watertight: yes
- Mean bbox relative error: 0.456
- Volume relative error: 1.539
- Normalized Chamfer distance: 0.029

## ROCm Inference Evidence

- GPU: AMD Radeon Graphics
- HIP version: 7.12.60610-2bd1678d3d
- Average inference latency: 15.57 sec
- Average generated throughput: 28.97 tokens/sec
- Peak allocated VRAM: 4.49 GB

## Candidate Status

| candidate | status | selected stage | raw error | safe error |
|---:|---|---|---|---|
| 0 | failed | failed | OCP.OCP.StdFail.StdFail_NotDone | OCP.OCP.StdFail.StdFail_NotDone |
| 1 | success | raw | n/a | n/a |
| 2 | failed | failed | OCP.OCP.StdFail.StdFail_NotDone | OCP.OCP.StdFail.StdFail_NotDone |
| 3 | failed | failed | OCP.OCP.StdFail.StdFail_NotDone | ValueError |
| 4 | success | raw | n/a | n/a |

## Engineering Notes

- Some candidates failed in the CAD kernel. Inspect the error table before manual editing.
- The best candidate has a large volume error; check missing holes, bosses, and support features.

## Error Distribution

| error type | count |
|---|---:|
| OCP.OCP.StdFail.StdFail_NotDone | 5 |
| ValueError | 1 |