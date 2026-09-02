"""Run a no-training forward check through the optimized prepared-data loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .src.config import load_config, resolve_project_root
from .src.models import build_model
from .src.runner import create_loaders, resolve_device


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", default=None, type=Path)
    parser.add_argument("--model", default="tcn", choices=("tcn", "gru"))
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--num-workers", default=None, type=int)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    project_root = resolve_project_root(args.config, args.project_root)
    device_name = args.device if args.device != "auto" else config["training"].get("device", "auto")
    device = resolve_device(device_name)
    train_loader, validation_loader, test_loader = create_loaders(
        config,
        project_root,
        num_workers_override=args.num_workers,
    )
    batch = next(iter(train_loader))
    model = build_model(args.model, config).to(device).eval()
    with torch.no_grad():
        outputs = model(
            batch["features"].to(device=device, dtype=torch.float32),
            time_mask=batch["time_mask"].to(device),
        )
    if not outputs or not all(bool(torch.isfinite(value).all()) for value in outputs.values()):
        raise RuntimeError("Runtime forward check produced non-finite model outputs")
    valid_lengths = batch["time_mask"].sum(dim=1).tolist()
    result = {
        "model": args.model,
        "device": str(device),
        "num_workers": args.num_workers,
        "batch_shape": list(batch["features"].shape),
        "valid_length_min": min(valid_lengths),
        "valid_length_max": max(valid_lengths),
        "loader_sizes": {
            "train": len(train_loader.dataset),
            "validation": len(validation_loader.dataset),
            "test": len(test_loader.dataset),
        },
        "output_shapes": {key: list(value.shape) for key, value in outputs.items()},
        "finite_outputs": True,
        "training_performed": False,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
