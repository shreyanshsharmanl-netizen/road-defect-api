"""Score the U-Net and baseline on the held-out test images."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import baseline
from .data import load_pairs, read_mask, split
from .inference import Inspector
from .metrics import f1_iou


def summarise(rows: list[dict]) -> dict:
    return {k: round(float(np.mean([r[k] for r in rows])), 4) for k in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/crackforest"))
    parser.add_argument("--weights", type=Path, default=Path("models/unet_crack.pt"))
    args = parser.parse_args()

    inspector = Inspector(args.weights)
    test = split(load_pairs(args.data))["test"]
    rows = {"unet_exact": [], "unet_tol2": [], "baseline_exact": [], "baseline_tol2": []}
    latency = []
    for image_path, label_path in test:
        image = Image.open(image_path)
        truth = read_mask(label_path).astype(bool)
        started = time.perf_counter()
        result = inspector.inspect(image)
        latency.append((time.perf_counter() - started) * 1000)
        base = baseline.predict(np.array(image.convert("RGB")))
        rows["unet_exact"].append(f1_iou(result.mask, truth))
        rows["unet_tol2"].append(f1_iou(result.mask, truth, tolerance=2))
        rows["baseline_exact"].append(f1_iou(base, truth))
        rows["baseline_tol2"].append(f1_iou(base, truth, tolerance=2))

    report = {
        "dataset": "CrackForest",
        "test_images": len(test),
        "image_size": "480x320",
        "metrics": {name: summarise(values) for name, values in rows.items()},
        "cpu_latency_ms_median": round(statistics.median(latency[1:]), 1),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/evaluation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
