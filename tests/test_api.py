import base64
import io

import pytest
from PIL import Image


def jpeg(size=(120, 90)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buffer, "JPEG")
    return buffer.getvalue()


def post(client, payload, name="road.jpg"):
    return client.post("/v1/inspect", files={"file": (name, payload)})


def test_health_ready(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["model_version"] == "test"


def test_inspect_returns_report_and_overlay_of_same_size(client):
    response = post(client, jpeg())
    body = response.json()
    assert response.status_code == 200
    assert body["grade"] in {"good", "fair", "poor", "severe"}
    assert 0 <= body["crack_percent"] <= 100
    assert Image.open(io.BytesIO(base64.b64decode(body["overlay_png_base64"]))).size == (120, 90)
    assert "X-Request-ID" in response.headers


def test_rejects_non_image(client):
    assert post(client, b"not an image").status_code == 415


def test_rejects_unsupported_format(client):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buffer, "BMP")
    assert post(client, buffer.getvalue()).status_code == 415


def test_rejects_truncated_image(client):
    assert post(client, jpeg()[:200]).status_code == 415


def test_rejects_oversized_upload(client):
    assert post(client, b"0" * (8 * 1024 * 1024 + 1)).status_code == 413


@pytest.mark.parametrize("size", [(9, 9), (1, 1), (1500, 700)])
def test_handles_awkward_sizes(client, size):
    assert post(client, jpeg(size)).status_code == 200


def test_metrics_counts_requests(client):
    post(client, jpeg())
    assert "roadscan_requests_total" in client.get("/metrics").text


def test_not_ready_without_model(monkeypatch):
    from fastapi.testclient import TestClient

    from roadscan.api import app

    monkeypatch.setenv("MODEL_PATH", "/nonexistent.pt")
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert post(client, jpeg()).status_code == 503
