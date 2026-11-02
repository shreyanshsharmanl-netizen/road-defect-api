import pytest
import torch
from fastapi.testclient import TestClient

from roadscan.api import app
from roadscan.model import UNet


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Service with random weights, so tests do not need the trained file."""
    weights = tmp_path / "model.pt"
    torch.save({"state": UNet().state_dict(), "version": "test"}, weights)
    monkeypatch.setenv("MODEL_PATH", str(weights))
    with TestClient(app) as test_client:
        yield test_client
