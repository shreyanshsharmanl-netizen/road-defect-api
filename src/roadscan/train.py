"""Train the U-Net and keep the checkpoint with the best validation F1."""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import __version__
from .data import CrackDataset, load_pairs, split
from .metrics import f1_iou
from .model import UNet


def dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, pos_weight=torch.tensor(8.0)
    )
    prob = torch.sigmoid(logits)
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice


@torch.no_grad()
def validate(model: UNet, loader: DataLoader, threshold: float = 0.5) -> float:
    model.eval()
    scores = []
    for image, mask in loader:
        pred = torch.sigmoid(model(image))[0, 0].numpy() > threshold
        scores.append(f1_iou(pred, mask[0, 0].numpy(), tolerance=2)["f1"])
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/crackforest"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--out", type=Path, default=Path("models/unet_crack.pt"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    parts = split(load_pairs(args.data))
    train_loader = DataLoader(
        CrackDataset(parts["train"], train=True, repeats=4), batch_size=8, shuffle=True
    )
    val_loader = DataLoader(CrackDataset(parts["val"]), batch_size=1)

    model = UNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=args.epochs * len(train_loader)
    )

    best, history, started = -1.0, [], time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, mask in train_loader:
            optimizer.zero_grad()
            loss = dice_bce(model(image), mask)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
        val_f1 = validate(model, val_loader)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_f1": val_f1})
        print(f"epoch {epoch:3d} loss {np.mean(losses):.4f} val_f1 {val_f1:.4f}", flush=True)
        if val_f1 > best:
            best = val_f1
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state": model.state_dict(), "version": __version__}, args.out)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/training.json").write_text(
        json.dumps(
            {"seed": args.seed, "best_val_f1": best, "seconds": time.time() - started,
             "history": history},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
