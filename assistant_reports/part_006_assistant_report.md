# Local CAD Assistant Report: part_006

## Task Summary

- Input views: `examples/part_006/views/view_0.png` ... `view_7.png`
- Generated candidates: 5
- Successful CAD exports: 3/5
- Raw successes: 3
- Safe repair successes: 0
- Failed candidates: 2
- Best candidate: `candidate_2`
- Best STEP: `outputs/part_006/candidate_2.step`
- Best STL: `outputs/part_006/candidate_2.stl`

## Geometry Quality

- Watertight: yes
- Mean bbox relative error: 0.303
- Volume relative error: 1.266
- Normalized Chamfer distance: 0.016

## ROCm Inference Evidence

- GPU: AMD Radeon Graphics
- HIP version: 7.12.60610-2bd1678d3d
- Average inference latency: 15.17 sec
- Average generated throughput: 26.33 tokens/sec
- Peak allocated VRAM: 4.49 GB

## Candidate Status

| candidate | status | selected stage | raw error | safe error |
|---:|---|---|---|---|
| 0 | success | raw | n/a | n/a |
| 1 | failed | failed | OCP.OCP.Standard.Standard_Failure | ValueError |
| 2 | success | raw | n/a | n/a |
| 3 | success | raw | n/a | n/a |
| 4 | failed | failed | OCP.OCP.StdFail.StdFail_NotDone | OCP.OCP.StdFail.StdFail_NotDone |

## Engineering Notes

- Some candidates failed in the CAD kernel. Inspect the error table before manual editing.
- The best candidate has a large volume error; check missing holes, bosses, and support features.

## Error Distribution

| error type | count |
|---|---:|
| ValueError | 2 |
| OCP.OCP.StdFail.StdFail_NotDone | 2 |
| OCP.OCP.Standard.Standard_Failure | 1 |