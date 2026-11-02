# Runbook

## Deploy
Pushes to `main` redeploy on Render through `render.yaml`. The service must pass `/readyz`.

## Roll back
Use Render's rollback to the previous deploy, or redeploy the previous `vX.Y.Z` image from GHCR.

## Signals
| Signal | Source | Action |
|--------|--------|--------|
| `/readyz` returns 503 | Health check | Model file missing. Check the `models/` copy step in the image build. |
| `roadscan_inference_seconds` rising | `/metrics` | Check CPU limits and image sizes. Scale out processes. |
| `roadscan_requests_total{outcome="rejected"}` rising | `/metrics` | Inspect logs by request ID for a misbehaving client. |

## Releasing a model
1. Run `make train` and `make evaluate`.
2. Compare `reports/evaluation.json` to the current version in a pull request.
3. Merge, update `CHANGELOG.md`, and tag `vX.Y.Z`.
