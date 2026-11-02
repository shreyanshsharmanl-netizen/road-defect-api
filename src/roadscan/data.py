"""CrackForest loading, deterministic splits and augmentation."""

from pathlib import Path

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset

MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.25, 0.25, 0.25], dtype=np.float32)


def load_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Return (image, label) paths. CrackForest has more images than labels."""
    pairs = []
    for label in sorted((root / "groundTruth").glob("*.mat")):
        image = root / "image" / f"{label.stem}.jpg"
        if image.exists():
            pairs.append((image, label))
    if not pairs:
        raise FileNotFoundError(f"No image/label pairs under {root}")
    return pairs


def split(pairs: list, seed: int = 42) -> dict[str, list]:
    """70/15/15 split. Fixed seed so every run scores the same test images."""
    order = np.random.default_rng(seed).permutation(len(pairs))
    n_val = n_test = round(len(pairs) * 0.15)
    return {
        "test": [pairs[i] for i in order[:n_test]],
        "val": [pairs[i] for i in order[n_test : n_test + n_val]],
        "train": [pairs[i] for i in order[n_test + n_val :]],
    }


def read_mask(path: Path) -> np.ndarray:
    seg = scipy.io.loadmat(path)["groundTruth"][0, 0]["Segmentation"]
    return (seg == 2).astype(np.float32)  # 1 = background, 2 = crack


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB to a normalised CHW tensor."""
    scaled = image.astype(np.float32) / 255.0
    return torch.from_numpy(((scaled - MEAN) / STD).transpose(2, 0, 1))


class CrackDataset(Dataset):
    def __init__(self, pairs: list, train: bool = False, crop: int = 192, repeats: int = 1):
        self.pairs, self.train, self.crop, self.repeats = pairs, train, crop, repeats

    def __len__(self) -> int:
        return len(self.pairs) * self.repeats

    def __getitem__(self, idx: int):
        image_path, label_path = self.pairs[idx % len(self.pairs)]
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = read_mask(label_path)
        if self.train:
            image, mask = self._augment(image, mask)
        else:
            h, w = (s - s % 8 for s in mask.shape)
            image, mask = image[:h, :w], mask[:h, :w]
        return to_tensor(image), torch.from_numpy(mask.copy())[None]

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        h, w = mask.shape
        # Cracks are about 1-2% of pixels, so prefer crops that contain some.
        for _ in range(8):
            y = np.random.randint(0, h - self.crop + 1)
            x = np.random.randint(0, w - self.crop + 1)
            if mask[y : y + self.crop, x : x + self.crop].sum() > 20:
                break
        image = image[y : y + self.crop, x : x + self.crop]
        mask = mask[y : y + self.crop, x : x + self.crop]
        if np.random.random() < 0.5:
            image, mask = image[:, ::-1], mask[:, ::-1]
        if np.random.random() < 0.5:
            image, mask = image[::-1], mask[::-1]
        k = np.random.randint(4)
        image, mask = np.rot90(image, k), np.rot90(mask, k)
        gain = np.random.uniform(0.8, 1.2)
        image = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return image.copy(), mask.copy()
