"""Training and validation loops shared by the command-line entry points."""

from __future__ import annotations

from collections import defaultdict
import fnmatch
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data.dataset import (
    LengthBucketBatchSampler,
    PreparedWindowDataset,
    load_prepared_metadata,
    prepared_collate,
    prepared_split_path,
)
from .evaluation import aggregate_sequence_predictions
from .losses import LossWeights, compute_loss, positive_weight_from_binary
from .metrics import composite_score, compute_metrics
from .models import build_model, parameter_count
from .provenance import git_commit, sha256_file, sha256_value
from .reporting import atomic_json_write


def resolve_device(requested: str = "auto") -> torch.device:
    requested = requested.lower()
    if requested != "auto":
        if requested == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if requested == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if requested == "cpu":
            return torch.device("cpu")
        raise RuntimeError(f"Requested device {requested!r} is unavailable")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_loaders(
    config: Config,
    project_root: Path,
    num_workers_override: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    processed_root = project_root / config["data"].get(
        "processed_root",
        f"{config['project'].get('data_root', 'data')}/processed",
    )
    batch_size = int(config["training"]["batch_size"])
    workers = int(
        config["training"].get("num_workers", 0)
        if num_workers_override is None
        else num_workers_override
    )
    if workers < 0:
        raise ValueError("num_workers must be non-negative")
    training_config = config["training"]
    label_policy = config["data"].get("label_policy", {})
    allow_weak_multi_rep_phase = bool(
        label_policy.get("allow_weak_phase_on_multi_rep_recordings", False)
    )
    datasets = {
        split: PreparedWindowDataset(
            prepared_split_path(processed_root, split),
            validate_features=False,
            return_numpy=True,
            allow_weak_phase_on_multi_rep_recordings=allow_weak_multi_rep_phase,
        )
        for split in ("train", "validation", "test")
    }

    def make_loader(split: str) -> DataLoader:
        sampler = LengthBucketBatchSampler(
            datasets[split],
            batch_size,
            shuffle=split == "train",
            seed=int(config["project"].get("seed", 42)),
        )
        kwargs: dict[str, Any] = {
            "batch_sampler": sampler,
            "collate_fn": prepared_collate,
            "num_workers": workers,
            "pin_memory": False,
        }
        if workers > 0:
            kwargs["persistent_workers"] = bool(training_config.get("persistent_workers", True))
            kwargs["prefetch_factor"] = int(training_config.get("prefetch_factor", 2))
        return DataLoader(datasets[split], **kwargs)

    loaders = {split: make_loader(split) for split in ("train", "validation", "test")}
    return loaders["train"], loaders["validation"], loaders["test"]


def _class_weights(
    values: np.ndarray,
    classes: int,
    sample_weights: np.ndarray | None = None,
) -> torch.Tensor:
    valid = values >= 0
    selected = values[valid].astype(np.int64)
    weights = sample_weights[valid] if sample_weights is not None else None
    counts = np.bincount(selected, minlength=classes).astype(np.float32)
    if weights is not None:
        counts = np.bincount(selected, minlength=classes, weights=weights).astype(np.float32)
    present = counts > 0.0
    class_weights = np.zeros(classes, dtype=np.float32)
    if present.any():
        class_weights[present] = counts[present].sum() / (
            float(present.sum()) * counts[present]
        )
        class_weights[present] /= max(float(class_weights[present].mean()), 1e-6)
        class_weights[present] = np.clip(class_weights[present], 0.25, 4.0)
    # An absent class must not dominate normalization or contribute loss.
    # This matters for phase.v1, where ``unknown`` can be absent while hold is
    # rare but valid; treating the absent class as one sample suppresses hold.
    return torch.from_numpy(class_weights)


def _loss_support(dataset: PreparedWindowDataset, config: Config, device: torch.device) -> dict[str, torch.Tensor]:
    label_weights = np.asarray(dataset.label_weights, dtype=np.float32)
    boundary_mask = (
        dataset.time_mask[..., None]
        & (dataset.boundary >= 0)
        & (label_weights[:, 2, None, None] > 0)
    )
    boundary_positive_values = torch.from_numpy(dataset.boundary[boundary_mask])
    boundary_positive = positive_weight_from_binary(
        boundary_positive_values,
        torch.ones_like(boundary_positive_values, dtype=torch.bool),
        max_weight=float(config["training"].get("boundary_positive_weight_cap", 10.0)),
    ).to(device)
    # Label arrays may be read-only memmaps; copy these small arrays before
    # converting them to tensors. Feature rows are copied in __getitem__.
    quality_values = torch.from_numpy(np.asarray(dataset.quality, dtype=np.float32).copy())
    quality_mask_array = dataset.quality_mask & (label_weights[:, 3, None] > 0)
    quality_mask = torch.from_numpy(np.asarray(quality_mask_array, dtype=bool).copy())
    phase_mask = (
        dataset.time_mask
        & dataset.phase_available[:, None]
        & (label_weights[:, 1, None] > 0)
    )
    phase_sample_weights = np.broadcast_to(label_weights[:, 1, None], dataset.phase.shape)
    quality_positive = torch.stack(
        [positive_weight_from_binary(quality_values[:, index], quality_mask[:, index]) for index in range(quality_values.shape[1])]
    ).to(device)
    expert_quality = np.asarray(dataset.expert_quality, dtype=np.int64)
    expert_quality_mask = np.asarray(dataset.expert_quality_mask, dtype=bool)
    expert_label_weight = np.asarray(dataset.expert_label_weight, dtype=np.float32)
    expert_valid = expert_quality_mask & (expert_quality >= 0) & (expert_label_weight > 0)
    expert_class_weights = _class_weights(
        expert_quality[expert_valid],
        int(config["model"].get("expert_quality_classes", 0) or 5),
        expert_label_weight[expert_valid],
    ).to(device)
    return {
        "family_class_weights": _class_weights(
            dataset.family,
            int(config["model"]["family_classes"]),
            label_weights[:, 0],
        ).to(device),
        "phase_class_weights": _class_weights(
            dataset.phase[phase_mask],
            int(config["model"]["phase_classes"]),
            phase_sample_weights[phase_mask],
        ).to(device),
        "boundary_positive_weight": boundary_positive,
        "quality_positive_weights": quality_positive,
        "expert_quality_class_weights": expert_class_weights,
    }


def _batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        if key == "sample_index":
            result[key] = value
        elif key == "features":
            result[key] = value.to(device=device, dtype=torch.float32)
        else:
            result[key] = value.to(device)
    return result


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    loss_weights: LossWeights,
    loss_support: dict[str, torch.Tensor],
    gradient_clip_norm: float = 1.0,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    # A frozen backbone must not update BatchNorm/dropout-style state during a
    # head-only experiment. The current models do not use BatchNorm, but this
    # invariant keeps future backbones safe and makes cached features valid.
    if training:
        for name, module in model.named_children():
            parameters = list(module.parameters())
            if parameters and not any(parameter.requires_grad for parameter in parameters):
                module.eval()
    totals: defaultdict[str, float] = defaultdict(float)
    batches = 0
    for batch in loader:
        batch = _batch_to_device(batch, device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(batch["features"], time_mask=batch["time_mask"])
            loss, values = compute_loss(outputs, batch, loss_weights, **loss_support)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
        for key, value in values.items():
            totals[key] += value
        batches += 1
    if batches == 0:
        raise RuntimeError("DataLoader produced no batches")
    return {key: value / batches for key, value in totals.items()}


def configure_trainable_parameters(model: torch.nn.Module, config: Config) -> list[str]:
    """Apply the explicit warm-start/freeze contract and return trainable names."""

    training = config["training"]
    for parameter in model.parameters():
        parameter.requires_grad = True

    freeze_backbone = bool(training.get("freeze_backbone", False))
    unfreeze_last_blocks = int(training.get("unfreeze_last_blocks", 0))
    if freeze_backbone or unfreeze_last_blocks:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("heads.")
        if unfreeze_last_blocks:
            if hasattr(model, "blocks"):
                blocks = getattr(model, "blocks")
                count = len(blocks)
                if unfreeze_last_blocks > count:
                    raise ValueError(
                        f"unfreeze_last_blocks={unfreeze_last_blocks} exceeds TCN block count {count}"
                    )
                first = count - unfreeze_last_blocks
                for name, parameter in model.named_parameters():
                    if name.startswith("blocks."):
                        index = int(name.split(".", 2)[1])
                        parameter.requires_grad = index >= first
            elif hasattr(model, "gru"):
                for parameter in getattr(model, "gru").parameters():
                    parameter.requires_grad = True

    patterns = training.get("trainable_patterns")
    if patterns is not None:
        pattern_values = [str(value) for value in patterns]
        for name, parameter in model.named_parameters():
            parameter.requires_grad = any(fnmatch.fnmatch(name, pattern) for pattern in pattern_values)

    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("training configuration leaves no trainable parameters")
    return trainable


def _optimizer_for_model(model: torch.nn.Module, config: Config) -> torch.optim.Optimizer:
    """Build optimizer parameter groups with separate head/backbone rates."""

    training = config["training"]
    default_rate = float(training["learning_rate"])
    head_rate = float(training.get("head_learning_rate", default_rate))
    backbone_rate = float(training.get("backbone_learning_rate", default_rate))
    head_parameters = []
    backbone_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (head_parameters if name.startswith("heads.") else backbone_parameters).append(parameter)
    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": backbone_rate})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_rate})
    if not groups:
        raise ValueError("optimizer received no trainable parameters")
    return torch.optim.AdamW(
        groups,
        weight_decay=float(training["weight_decay"]),
    )


def _move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device)


def _restore_rng_state(checkpoint: dict[str, Any]) -> None:
    """Restore all RNG streams captured by an exact-resume checkpoint."""

    torch_state = checkpoint.get("rng_state_torch")
    if torch_state is not None:
        # Checkpoints loaded with ``map_location=mps`` carry the CPU RNG byte
        # tensor onto MPS. ``torch.set_rng_state`` only accepts a CPU byte
        # tensor, so normalize the device and dtype before restoring it.
        if isinstance(torch_state, torch.Tensor):
            torch_state = torch_state.detach().to(device="cpu", dtype=torch.uint8)
        else:
            torch_state = torch.as_tensor(torch_state, dtype=torch.uint8, device="cpu")
        torch.set_rng_state(torch_state)
    numpy_state = checkpoint.get("rng_state_numpy")
    if numpy_state is not None:
        np.random.set_state(numpy_state)
    python_state = checkpoint.get("rng_state_python")
    if python_state is not None:
        random.setstate(python_state)


def _resolve_checkpoint_path(raw_path: str | None, project_root: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def _load_model_checkpoint(
    path: Path,
    model: torch.nn.Module,
    config: Config,
    device: torch.device,
    expected_model_name: str | None = None,
    allow_normalization_mismatch: bool = False,
) -> dict[str, Any]:
    from .config import validate_checkpoint_compatibility

    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    validate_checkpoint_compatibility(
        checkpoint,
        config,
        allow_normalization_mismatch=allow_normalization_mismatch,
    )
    if checkpoint.get("model_name") not in {"tcn", "gru"}:
        raise ValueError("checkpoint model_name is unsupported")
    if expected_model_name is not None and checkpoint.get("model_name") != expected_model_name:
        raise ValueError(
            f"checkpoint model_name={checkpoint.get('model_name')!r} does not match "
            f"requested model={expected_model_name!r}"
        )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint


def _checkpoint_payload(
    *,
    model: torch.nn.Module,
    model_name: str,
    config: Config,
    seed: int,
    device: torch.device,
    epoch: int,
    best_score: float,
    best_epoch: int,
    stale_epochs: int,
    history: list[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    trainable_layers: list[str],
    parent_checkpoint: str | None,
    parent_checkpoint_hash: str | None,
    parent_checkpoint_normalization_version: str | None,
    parent_normalization_transfer: bool,
    train_seconds_total: float,
    validation_seconds_total: float,
    project_root: Path,
    sampler_epoch: int | None,
) -> dict[str, Any]:
    return {
        "checkpoint_schema_version": "adaptfit.checkpoint.v2",
        "model_name": model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "seed": seed,
        "device": str(device),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": parameter_count(model),
        "epoch": epoch,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "stale_epochs": stale_epochs,
        "history": history,
        "trainable_layers": trainable_layers,
        "parent_checkpoint": parent_checkpoint,
        "parent_checkpoint_hash": parent_checkpoint_hash,
        "parent_checkpoint_normalization_version": parent_checkpoint_normalization_version,
        "parent_normalization_transfer": parent_normalization_transfer,
        "rng_state_torch": torch.get_rng_state(),
        "rng_state_numpy": np.random.get_state(),
        "rng_state_python": random.getstate(),
        "train_seconds_total": train_seconds_total,
        "validation_seconds_total": validation_seconds_total,
        "sampler_epoch": sampler_epoch,
        "source_commit": git_commit(project_root),
    }


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    model.eval()
    output_values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    target_values: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    sample_indices: list[np.ndarray] = []
    frame_count = int(getattr(loader.dataset, "features").shape[1])

    def pad_time_axis(values: np.ndarray, fill_value: float | int | bool) -> np.ndarray:
        if values.ndim < 2 or values.shape[1] == frame_count:
            return values
        if values.shape[1] > frame_count:
            raise ValueError("Collected batch has more frames than the prepared window")
        shape = (values.shape[0], frame_count, *values.shape[2:])
        padded = np.full(shape, fill_value, dtype=values.dtype)
        padded[:, : values.shape[1]] = values
        return padded

    for batch in loader:
        batch_device = _batch_to_device(batch, device)
        outputs = model(batch_device["features"], time_mask=batch_device["time_mask"])
        for key, value in outputs.items():
            array = value.detach().cpu().numpy()
            output_values[key].append(
                pad_time_axis(array, 0.0) if key in {"phase_logits", "boundary_logits", "tracking_logits"} else array
            )
        for key in (
            "family",
            "phase",
            "boundary",
            "quality",
            "quality_mask",
            "expert_quality",
            "expert_quality_mask",
            "expert_label_weight",
            "tracking_target",
            "time_mask",
        ):
            if key in batch:
                array = batch[key].detach().cpu().numpy()
                fill_value: float | int | bool = {
                    "phase": -1,
                    "boundary": -1.0,
                    "tracking_target": -1.0,
                    "time_mask": False,
                }.get(key, 0.0)
                target_values[key].append(
                    pad_time_axis(array, fill_value)
                    if key in {"phase", "boundary", "tracking_target", "time_mask"}
                    else array
                )
        if "sample_index" in batch:
            sample_indices.append(batch["sample_index"].detach().cpu().numpy())

    outputs = {key: np.concatenate(values, axis=0) for key, values in output_values.items()}
    targets = {key: np.concatenate(values, axis=0) for key, values in target_values.items()}
    if sample_indices:
        order = np.argsort(np.concatenate(sample_indices, axis=0), kind="stable")
        outputs = {key: value[order] for key, value in outputs.items()}
        targets = {key: value[order] for key, value in targets.items()}
    return outputs, targets


def train_model(
    model_name: str,
    config: Config,
    project_root: Path,
    seed: int,
    requested_device: str,
) -> dict[str, Any]:
    seed_everything(seed)
    device = resolve_device(requested_device)
    train_loader, validation_loader, test_loader = create_loaders(config, project_root)
    train_dataset: PreparedWindowDataset = train_loader.dataset  # type: ignore[assignment]
    processed_root = project_root / config["data"].get(
        "processed_root",
        f"{config['project'].get('data_root', 'data')}/processed",
    )
    try:
        validation_metadata: list[dict[str, Any]] | None = load_prepared_metadata(
            processed_root,
            "validation",
            expected_count=len(validation_loader.dataset),
        )
    except FileNotFoundError:
        # Older NPZ-only prepared runs do not have JSONL metadata. They can
        # still be trained, but early stopping must explicitly fall back to
        # window-level validation because sequence reconstruction is unsafe.
        validation_metadata = None
        print(
            "warning: validation metadata is missing; falling back to "
            "window-level validation scoring"
        )
    model = build_model(model_name, config).to(device)
    trainable_layers = configure_trainable_parameters(model, config)
    optimizer = _optimizer_for_model(model, config)
    training_config = config["training"]
    init_path = _resolve_checkpoint_path(training_config.get("init_checkpoint"), project_root)
    resume_path = _resolve_checkpoint_path(training_config.get("resume_checkpoint"), project_root)
    parent_checkpoint: str | None = None
    parent_checkpoint_hash: str | None = None
    parent_checkpoint_normalization_version: str | None = None
    parent_normalization_transfer = False
    resumed_checkpoint: dict[str, Any] | None = None
    if init_path is not None:
        parent_payload = _load_model_checkpoint(
            init_path,
            model,
            config,
            device,
            expected_model_name=model_name,
            allow_normalization_mismatch=bool(
                training_config.get("allow_parent_normalization_transfer", False)
            ),
        )
        parent_checkpoint = str(init_path)
        parent_checkpoint_hash = sha256_file(init_path)
        parent_checkpoint_normalization_version = parent_payload.get("config", {}).get(
            "data", {}
        ).get("normalization_version")
        parent_normalization_transfer = (
            parent_checkpoint_normalization_version
            != config["data"].get("normalization_version")
        )
    if resume_path is not None:
        resumed_checkpoint = _load_model_checkpoint(
            resume_path,
            model,
            config,
            device,
            expected_model_name=model_name,
        )
        if resumed_checkpoint.get("optimizer_state_dict") is None:
            raise ValueError("exact resume requires optimizer_state_dict in the checkpoint")
        optimizer.load_state_dict(resumed_checkpoint["optimizer_state_dict"])
        _move_optimizer_state(optimizer, device)
        saved_layers = resumed_checkpoint.get("trainable_layers")
        if saved_layers is not None and list(saved_layers) != trainable_layers:
            raise ValueError("resume checkpoint trainable layers do not match the requested configuration")
        parent_checkpoint = str(resume_path)
        parent_checkpoint_hash = sha256_file(resume_path)
        parent_checkpoint_normalization_version = resumed_checkpoint.get(
            "parent_checkpoint_normalization_version",
            resumed_checkpoint.get("config", {}).get("data", {}).get("normalization_version"),
        )
        parent_normalization_transfer = bool(
            resumed_checkpoint.get("parent_normalization_transfer", False)
        )
        _restore_rng_state(resumed_checkpoint)
        saved_sampler_epoch = resumed_checkpoint.get("sampler_epoch")
        if saved_sampler_epoch is not None and hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(int(saved_sampler_epoch))
    loss_weights = LossWeights(**{
        key: float(value)
        for key, value in config["training"]["loss_weights"].items()
    })
    loss_support = _loss_support(train_dataset, config, device)
    max_epochs = int(training_config["max_epochs"])
    patience = int(training_config["early_stopping_patience"])
    validation_interval = int(training_config.get("validation_interval", 1))
    validate_on_first = bool(training_config.get("validate_on_first_epoch", True))
    validate_on_final = bool(training_config.get("validate_on_final_epoch", True))
    gradient_clip_norm = float(training_config.get("gradient_clip_norm", 1.0))
    checkpoint_root = project_root / config["project"].get("artifacts_root", "artifacts") / "checkpoints"
    metrics_root = project_root / config["project"].get("artifacts_root", "artifacts") / "metrics"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)
    checkpoint_filename = "gru_baseline.pt" if model_name == "gru" else f"{model_name}_best.pt"
    checkpoint_path = checkpoint_root / checkpoint_filename
    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    validation_checks = 0
    train_seconds_total = 0.0
    validation_seconds_total = 0.0
    start_epoch = 1
    if resumed_checkpoint is not None:
        start_epoch = int(resumed_checkpoint.get("epoch", 0)) + 1
        best_score = float(resumed_checkpoint.get("best_score", resumed_checkpoint.get("validation_score", float("-inf"))))
        best_epoch = int(resumed_checkpoint.get("best_epoch", 0))
        stale_epochs = int(resumed_checkpoint.get("stale_epochs", 0))
        history = list(resumed_checkpoint.get("history", []))
        validation_checks = sum(1 for row in history if not row.get("validation_skipped", False))
        train_seconds_total = float(resumed_checkpoint.get("train_seconds_total", 0.0))
        validation_seconds_total = float(resumed_checkpoint.get("validation_seconds_total", 0.0))

    latest_checkpoint_path = checkpoint_root / (
        "gru_latest.pt" if model_name == "gru" else f"{model_name}_latest.pt"
    )

    def save_latest_checkpoint() -> None:
        if not bool(training_config.get("save_latest", True)):
            return
        torch.save(
            _checkpoint_payload(
                model=model,
                model_name=model_name,
                config=config,
                seed=seed,
                device=device,
                epoch=history[-1]["epoch"] if history else 0,
                best_score=best_score,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                history=history,
                optimizer=optimizer,
                trainable_layers=trainable_layers,
                parent_checkpoint=parent_checkpoint,
                parent_checkpoint_hash=parent_checkpoint_hash,
                parent_checkpoint_normalization_version=parent_checkpoint_normalization_version,
                parent_normalization_transfer=parent_normalization_transfer,
                train_seconds_total=train_seconds_total,
                validation_seconds_total=validation_seconds_total,
                project_root=project_root,
                sampler_epoch=getattr(train_loader.batch_sampler, "epoch", None),
            ),
            latest_checkpoint_path,
        )

    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = parameter_count(model)
    print(
        f"[{model_name}] device={device} parameters={total_parameter_count:,} "
        f"trainable={trainable_parameter_count:,}"
    )
    for epoch in range(start_epoch, max_epochs + 1):
        train_started = time.perf_counter()
        train_loss = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_weights,
            loss_support,
            gradient_clip_norm=gradient_clip_norm,
        )
        train_seconds = time.perf_counter() - train_started
        train_seconds_total += train_seconds
        should_validate = (
            (epoch == 1 and validate_on_first)
            or epoch % validation_interval == 0
            or (epoch == max_epochs and validate_on_final)
        )
        if not should_validate:
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_seconds": train_seconds,
                    "train_samples_per_second": len(train_loader.dataset) / max(train_seconds, 1e-9),
                    "validation_skipped": True,
                    "validation_metrics": None,
                    "validation_sequence_metrics": None,
                    "validation_window_score": None,
                    "validation_sequence_score": None,
                    "validation_score": None,
                }
            )
            print(
                f"[{model_name}] epoch={epoch:03d} loss={train_loss['total']:.4f} "
                f"validation=skipped train_sps={len(train_loader.dataset) / max(train_seconds, 1e-9):.1f}"
            )
            save_latest_checkpoint()
            continue

        validation_started = time.perf_counter()
        validation_outputs, validation_targets = collect_predictions(model, validation_loader, device)
        validation_metrics = compute_metrics(validation_outputs, validation_targets)
        validation_sequence_metrics: dict[str, float] | None = None
        if validation_metadata is not None:
            validation_sequence_outputs, validation_sequence_targets, _ = aggregate_sequence_predictions(
                validation_outputs,
                validation_targets,
                validation_metadata,
            )
            validation_sequence_metrics = compute_metrics(
                validation_sequence_outputs,
                validation_sequence_targets,
            )
        validation_window_score = composite_score(validation_metrics)
        validation_sequence_score = (
            composite_score(validation_sequence_metrics)
            if validation_sequence_metrics is not None
            else float("nan")
        )
        score = (
            validation_sequence_score
            if np.isfinite(validation_sequence_score)
            else validation_window_score
        )
        validation_seconds = time.perf_counter() - validation_started
        validation_seconds_total += validation_seconds
        validation_checks += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seconds": train_seconds,
            "train_samples_per_second": len(train_loader.dataset) / max(train_seconds, 1e-9),
            "validation_seconds": validation_seconds,
            "validation_check": validation_checks,
            "validation_skipped": False,
            "validation_metrics": validation_metrics,
            "validation_sequence_metrics": validation_sequence_metrics,
            "validation_window_score": validation_window_score,
            "validation_sequence_score": validation_sequence_score,
            "validation_score": score,
        }
        history.append(row)
        print(
            f"[{model_name}] epoch={epoch:03d} loss={train_loss['total']:.4f} "
            f"val_sequence_score={score:.4f} phase_f1={validation_metrics['phase_macro_f1']:.4f} "
            f"boundary_f1={validation_metrics['boundary_f1']:.4f}"
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            best_payload = _checkpoint_payload(
                model=model,
                model_name=model_name,
                config=config,
                seed=seed,
                device=device,
                epoch=epoch,
                best_score=best_score,
                best_epoch=best_epoch,
                stale_epochs=stale_epochs,
                history=history,
                optimizer=optimizer,
                trainable_layers=trainable_layers,
                parent_checkpoint=parent_checkpoint,
                parent_checkpoint_hash=parent_checkpoint_hash,
                parent_checkpoint_normalization_version=parent_checkpoint_normalization_version,
                parent_normalization_transfer=parent_normalization_transfer,
                train_seconds_total=train_seconds_total,
                validation_seconds_total=validation_seconds_total,
                project_root=project_root,
                sampler_epoch=getattr(train_loader.batch_sampler, "epoch", None),
            )
            best_payload.update(
                {
                    "validation_metrics": validation_metrics,
                    "validation_sequence_metrics": validation_sequence_metrics,
                    "validation_window_score": validation_window_score,
                    "validation_sequence_score": validation_sequence_score,
                    "validation_score": score,
                }
            )
            torch.save(best_payload, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"[{model_name}] early stopping at epoch {epoch}")
                save_latest_checkpoint()
                break

        save_latest_checkpoint()

    if not checkpoint_path.exists():
        raise RuntimeError("training completed without producing a best checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics: dict[str, float] | None = None
    if bool(training_config.get("evaluate_test_after_training", True)):
        test_outputs, test_targets = collect_predictions(model, test_loader, device)
        test_metrics = compute_metrics(test_outputs, test_targets)
    result = {
        "model_name": model_name,
        "device": str(device),
        "parameter_count": total_parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "best_epoch": best_epoch,
        "best_validation_score": best_score,
        "test_metrics": test_metrics,
        "history": history,
        "checkpoint": str(checkpoint_path),
        "validation_checks": validation_checks,
        "train_seconds_total": train_seconds_total,
        "validation_seconds_total": validation_seconds_total,
        "phase_policy_reasons": dict(train_dataset.phase_policy_reasons),
        "trainable_layers": trainable_layers,
        "parent_checkpoint": parent_checkpoint,
        "parent_checkpoint_hash": parent_checkpoint_hash,
        "parent_checkpoint_normalization_version": parent_checkpoint_normalization_version,
        "parent_normalization_transfer": parent_normalization_transfer,
        "latest_checkpoint": str(latest_checkpoint_path) if latest_checkpoint_path.exists() else None,
        "test_evaluation_performed": test_metrics is not None,
        "config_hash": sha256_value(config),
    }
    atomic_json_write(metrics_root / f"{model_name}_history.json", result)
    return result
