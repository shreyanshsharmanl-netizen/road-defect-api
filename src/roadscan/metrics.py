"""Segmentation metrics, including the tolerance used in crack papers."""

import numpy as np
from scipy.ndimage import binary_dilation


def f1_iou(pred: np.ndarray, truth: np.ndarray, tolerance: int = 0) -> dict[str, float]:
    """Precision, recall, F1 and IoU for boolean masks.

    Crack annotations are hand-drawn and about a pixel wide, so a prediction
    next to the line is not really wrong. With tolerance > 0, a predicted pixel
    counts if a true pixel lies within that distance, and vice versa.
    """
    pred, truth = pred.astype(bool), truth.astype(bool)
    if tolerance:
        structure = np.ones((2 * tolerance + 1,) * 2, bool)
        precise = (pred & binary_dilation(truth, structure)).sum()
        covered = (truth & binary_dilation(pred, structure)).sum()
    else:
        precise = covered = (pred & truth).sum()
    precision = precise / pred.sum() if pred.sum() else float(not truth.any())
    recall = covered / truth.sum() if truth.sum() else float(not pred.any())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(f1 / (2 - f1)) if f1 else 0.0,
    }
