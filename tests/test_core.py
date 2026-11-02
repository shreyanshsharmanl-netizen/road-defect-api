import numpy as np
import pytest
import torch

from roadscan.baseline import predict
from roadscan.data import split
from roadscan.inference import grade
from roadscan.metrics import f1_iou
from roadscan.model import UNet


def test_perfect_prediction():
    truth = np.zeros((10, 10), bool)
    truth[4, :] = True
    assert f1_iou(truth, truth)["f1"] == 1.0


def test_both_empty_is_perfect_and_miss_is_zero():
    empty = np.zeros((5, 5), bool)
    full = np.ones((5, 5), bool)
    assert f1_iou(empty, empty)["f1"] == 1.0
    assert f1_iou(empty, full)["f1"] == 0.0


def test_tolerance_accepts_nearby_line():
    truth, pred = np.zeros((10, 10), bool), np.zeros((10, 10), bool)
    truth[4, :], pred[5, :] = True, True
    assert f1_iou(pred, truth)["f1"] == 0.0
    assert f1_iou(pred, truth, tolerance=1)["f1"] == 1.0


def test_split_is_deterministic_and_disjoint():
    items = list(range(118))
    first, second = split(items), split(items)
    assert first == second
    assert sum(len(v) for v in first.values()) == 118
    assert not set(first["train"]) & set(first["test"])


@pytest.mark.parametrize(
    "percent,expected", [(0, "good"), (0.5, "fair"), (2, "poor"), (5, "severe"), (40, "severe")]
)
def test_grade_boundaries(percent, expected):
    assert grade(percent) == expected


def test_unet_keeps_shape():
    assert UNet()(torch.zeros(1, 3, 64, 96)).shape == (1, 1, 64, 96)


def test_baseline_finds_dark_line():
    image = np.full((60, 60, 3), 150, np.uint8)
    image[30:33, :] = 40
    assert predict(image)[31, 30]
