# Test plan

| Level | What | Where | When |
|-------|------|-------|------|
| Unit | Metrics, split, grades, model shape, baseline | `tests/test_core.py` | Every push |
| API | Happy path, bad input, odd sizes, missing model | `tests/test_api.py` | Every push |
| Container | Image builds and reaches `/readyz` | CI `image` job | Every push |
| Model | Held-out test metrics against baseline | `make evaluate` | Each new model |
| Manual | Upload a sample in the browser, confirm the overlay lines up | Demo | Each release |

API tests use random weights so CI does not depend on training. Model quality is checked separately by `make evaluate`.
