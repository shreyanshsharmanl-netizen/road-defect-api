# Requirements

## Problem
Manual road surveys are slow and expensive. Inspectors need a first pass that flags which photos show cracking so they can prioritise review.

## Users
- **Road inspector:** uploads a photo and sees where the cracks are.
- **Developer:** calls an API from a survey pipeline.

## Functional requirements
| ID | Requirement | Verified by |
|----|-------------|-------------|
| F1 | Accept JPEG, PNG or WebP uploads | `test_inspect_returns_report_and_overlay_of_same_size` |
| F2 | Return crack area, region count, grade and an overlay | same |
| F3 | Reject bad, truncated, oversized or unsupported files with a clear 4xx | `test_rejects_*` |
| F4 | Provide a browser demo | Manual check, see `docs/test-plan.md` |

## Non-functional requirements
| ID | Requirement | Status |
|----|-------------|--------|
| N1 | Beat a classical image-processing baseline on held-out images | Measured in `reports/evaluation.json` |
| N2 | Run on CPU, with no GPU needed | Latency measured in `reports/evaluation.json` |
| N3 | Separate liveness and readiness checks | `/healthz`, `/readyz` |
| N4 | Expose request and latency metrics | `/metrics` |
| N5 | Container runs as a non-root user | `Dockerfile` |

## Out of scope and limits
- Grades are illustrative thresholds on crack area. They are not a Pavement Condition Index and must not drive maintenance decisions alone.
- There is no authentication or rate limiting in this demo. Both are required before public production use.
- The model has only seen CrackForest-style top-down photos. Other cameras and angles may perform worse.
