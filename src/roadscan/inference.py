"""Model loading and single-image prediction."""

import base64
import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label

from .data import to_tensor
from .model import UNet

MAX_SIDE = 1024
# Illustrative cut-offs on the share of pixels marked as crack. These are not
# an engineering standard such as PCI; see docs/requirements.md.
GRADES = [(0.5, "good"), (2.0, "fair"), (5.0, "poor")]


def grade(crack_percent: float) -> str:
    for limit, name in GRADES:
        if crack_percent < limit:
            return name
    return "severe"


@dataclass
class Inspection:
    crack_percent: float
    regions: int
    grade: str
    latency_ms: float
    mask: np.ndarray
    image: np.ndarray

    def overlay_png(self) -> str:
        out = self.image.copy()
        out[self.mask] = (0.35 * out[self.mask] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(out).save(buffer, "PNG")
        return base64.b64encode(buffer.getvalue()).decode()


class Inspector:
    def __init__(self, weights: Path, threshold: float = 0.5):
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True)
        self.model = UNet()
        self.model.load_state_dict(checkpoint["state"])
        self.model.eval()
        self.version = checkpoint.get("version", "unknown")
        self.threshold = threshold

    @torch.no_grad()
    def inspect(self, image: Image.Image) -> Inspection:
        started = time.perf_counter()
        image = image.convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        array = np.array(image)
        h, w = array.shape[:2]
        pad_h, pad_w = -h % 8, -w % 8
        padded = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        prob = torch.sigmoid(self.model(to_tensor(padded)[None]))[0, 0].numpy()[:h, :w]
        mask = prob > self.threshold
        percent = float(mask.mean() * 100)
        return Inspection(
            crack_percent=round(percent, 3),
            regions=int(label(mask)[1]),
            grade=grade(percent),
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            mask=mask,
            image=array,
        )
