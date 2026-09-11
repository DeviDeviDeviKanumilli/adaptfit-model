"""Configuration loading and path helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, TypedDict, cast

import yaml

from .features.anatomy import feature_dimensions


class ConfigError(ValueError):
    """Raised when a training configuration violates the v1 contract."""


class ProjectConfig(TypedDict, total=False):
    name: str
    experiment_id: str
    seed: int
    feature_schema_version: str
    decoder_version: str
    data_root: str
    artifacts_root: str
    parent_artifact_root: str


class SourceConfig(TypedDict, total=False):
    name: str
    path: str
    role: str
    license: str
    url: str
    archive_url: str
    enabled: bool
    required: bool
    allow_single_clip_boundaries: bool
    metadata_file: str
    pose_file: str


class SplitConfig(TypedDict, total=False):
    train: float
    validation: float
    test: float
    required_test_sources: list[str]
    required_test_positions: list[str]
    enforce_required_test_coverage: bool


class LabelPolicyConfig(TypedDict, total=False):
    allow_single_clip_boundaries: bool
    allow_procedural_quality: bool
    allow_uco_composite_quality: bool
    allow_weak_phase_on_multi_rep_recordings: bool


class DataConfig(TypedDict, total=False):
    target_fps: int
    num_joints: int
    processed_root: str
    window_frames: int
    window_stride: int
    max_interpolation_gap: int
    feature_storage_dtype: str
    normalization_version: str
    strict_source_paths: bool
    split: SplitConfig
    required_roles: list[str]
    sources: list[SourceConfig]
    synthetic: dict[str, Any]
    synthetic_seed: dict[str, Any]
    label_policy: LabelPolicyConfig


class FeatureConfig(TypedDict, total=False):
    input_dim: int
    continuous_dim: int
    include_diagnostics: bool
    angle_count: int
    profile_context_dim: int
    position_context_dim: int


class TCNConfig(TypedDict, total=False):
    channels: int
    kernel_size: int
    dilations: list[int]
    dropout: float


class GRUConfig(TypedDict, total=False):
    hidden_size: int
    num_layers: int


class ModelConfig(TypedDict, total=False):
    tcn: TCNConfig
    gru: GRUConfig
    family_classes: int
    phase_classes: int
    quality_outputs: int
    expert_quality_classes: int
    pooling: str


class LossWeightsConfig(TypedDict, total=False):
    family: float
    phase: float
    boundary: float
    quality: float
    expert_quality: float
    tracking: float


class TrainingConfig(TypedDict, total=False):
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    num_workers: int
    persistent_workers: bool
    prefetch_factor: int
    validation_interval: int
    validate_on_first_epoch: bool
    validate_on_final_epoch: bool
    device: str
    float32: bool
    head_learning_rate: float
    backbone_learning_rate: float
    boundary_positive_weight_cap: float
    init_checkpoint: str
    resume_checkpoint: str
    allow_parent_normalization_transfer: bool
    freeze_backbone: bool
    unfreeze_last_blocks: int
    trainable_patterns: list[str]
    evaluate_test_after_training: bool
    save_latest: bool
    loss_weights: LossWeightsConfig


class Config(TypedDict):
    project: ProjectConfig
    data: DataConfig
    features: FeatureConfig
    model: ModelConfig
    training: TrainingConfig


def validate_config(config: Mapping[str, Any]) -> Config:
    required_sections = ("project", "data", "features", "model", "training")
    missing = [section for section in required_sections if not isinstance(config.get(section), Mapping)]
    if missing:
        raise ConfigError(f"Config is missing mapping sections: {', '.join(missing)}")

    features = config["features"]
    include_diagnostics = features.get("include_diagnostics", False)
    if not isinstance(include_diagnostics, bool):
        raise ConfigError("features.include_diagnostics must be boolean")
    expected_feature_dim, expected_continuous_dim = feature_dimensions(include_diagnostics)
    if int(features.get("input_dim", -1)) != expected_feature_dim:
        raise ConfigError(
            f"features.input_dim must be {expected_feature_dim} for the selected feature schema"
        )
    if int(features.get("continuous_dim", -1)) != expected_continuous_dim:
        raise ConfigError(
            f"features.continuous_dim must be {expected_continuous_dim} for the selected feature schema"
        )

    data = config["data"]
    if int(data.get("num_joints", -1)) != 33:
        raise ConfigError("v1 requires data.num_joints=33")
    target_fps = float(data.get("target_fps", 0.0))
    if not math.isfinite(target_fps) or target_fps <= 0.0:
        raise ConfigError("data.target_fps must be finite and positive")
    if int(data.get("max_interpolation_gap", -1)) < 0:
        raise ConfigError("data.max_interpolation_gap must be non-negative")
    if int(data.get("window_frames", 0)) <= 0 or int(data.get("window_stride", 0)) <= 0:
        raise ConfigError("data.window_frames and data.window_stride must be positive")
    if data.get("feature_storage_dtype", "float32") not in {"float16", "float32"}:
        raise ConfigError("data.feature_storage_dtype must be 'float16' or 'float32'")
    if "normalization_version" in data and (
        not isinstance(data["normalization_version"], str)
        or not data["normalization_version"].strip()
    ):
        raise ConfigError("data.normalization_version must be a non-empty string")
    if "strict_source_paths" in data and not isinstance(data["strict_source_paths"], bool):
        raise ConfigError("data.strict_source_paths must be boolean")
    split = data.get("split", {})
    train_fraction = float(split.get("train", 0.0))
    validation_fraction = float(split.get("validation", 0.0))
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ConfigError("data.split must leave a positive test fraction")
    for name in ("required_test_sources", "required_test_positions"):
        if name in split and not isinstance(split[name], list):
            raise ConfigError(f"data.split.{name} must be a list")
        if name in split and any(not isinstance(value, str) or not value.strip() for value in split[name]):
            raise ConfigError(f"data.split.{name} must contain non-empty strings")
    if "enforce_required_test_coverage" in split and not isinstance(
        split["enforce_required_test_coverage"], bool
    ):
        raise ConfigError("data.split.enforce_required_test_coverage must be boolean")
    if not isinstance(data.get("sources"), list):
        raise ConfigError("data.sources must be a list")
    label_policy = data.get("label_policy", {})
    if not isinstance(label_policy, Mapping):
        raise ConfigError("data.label_policy must be a mapping")
    for name in (
        "allow_single_clip_boundaries",
        "allow_procedural_quality",
        "allow_weak_phase_on_multi_rep_recordings",
    ):
        if name in label_policy and not isinstance(label_policy[name], bool):
            raise ConfigError(f"data.label_policy.{name} must be boolean")
    if "allow_uco_composite_quality" in label_policy and not isinstance(
        label_policy["allow_uco_composite_quality"], bool
    ):
        raise ConfigError("data.label_policy.allow_uco_composite_quality must be boolean")

    model = config["model"]
    if int(model.get("family_classes", 0)) != 6:
        raise ConfigError("v1 requires model.family_classes=6")
    if int(model.get("phase_classes", 0)) != 5:
        raise ConfigError("v1 requires model.phase_classes=5")
    if int(model.get("quality_outputs", 0)) != 4:
        raise ConfigError("v1 requires model.quality_outputs=4")
    expert_quality_classes = int(model.get("expert_quality_classes", 0))
    if expert_quality_classes not in {0} and expert_quality_classes < 2:
        raise ConfigError("model.expert_quality_classes must be 0 or at least 2")
    if model.get("pooling", "last_valid") not in {"last_valid", "masked_mean"}:
        raise ConfigError("model.pooling must be 'last_valid' or 'masked_mean'")
    for name in ("tcn", "gru"):
        if not isinstance(model.get(name), Mapping):
            raise ConfigError(f"model.{name} must be a mapping")

    training = config["training"]
    if int(training.get("batch_size", 0)) <= 0 or int(training.get("max_epochs", 0)) <= 0:
        raise ConfigError("training.batch_size and training.max_epochs must be positive")
    if float(training.get("learning_rate", 0.0)) <= 0:
        raise ConfigError("training.learning_rate must be positive")
    if "float32" in training and not isinstance(training["float32"], bool):
        raise ConfigError("training.float32 must be boolean")
    if not bool(training.get("float32", True)):
        raise ConfigError("Only float32 model computation is supported in this release")
    if int(training.get("num_workers", 0)) < 0:
        raise ConfigError("training.num_workers must be non-negative")
    if int(training.get("prefetch_factor", 2)) <= 0:
        raise ConfigError("training.prefetch_factor must be positive")
    if int(training.get("validation_interval", 1)) <= 0:
        raise ConfigError("training.validation_interval must be positive")
    for name in ("head_learning_rate", "backbone_learning_rate"):
        if name in training:
            value = float(training[name])
            if not math.isfinite(value) or value <= 0.0:
                raise ConfigError(f"training.{name} must be finite and positive")
    if "boundary_positive_weight_cap" in training:
        boundary_cap = float(training["boundary_positive_weight_cap"])
        if not math.isfinite(boundary_cap) or boundary_cap < 1.0:
            raise ConfigError("training.boundary_positive_weight_cap must be finite and at least 1")
    for name in (
        "freeze_backbone",
        "evaluate_test_after_training",
        "save_latest",
        "allow_parent_normalization_transfer",
    ):
        if name in training and not isinstance(training[name], bool):
            raise ConfigError(f"training.{name} must be boolean")
    if "unfreeze_last_blocks" in training:
        try:
            unfreeze_last_blocks = int(training["unfreeze_last_blocks"])
        except (TypeError, ValueError) as error:
            raise ConfigError("training.unfreeze_last_blocks must be a non-negative integer") from error
        if unfreeze_last_blocks < 0:
            raise ConfigError("training.unfreeze_last_blocks must be a non-negative integer")
    for name in ("init_checkpoint", "resume_checkpoint"):
        if name in training and training[name] is not None and (
            not isinstance(training[name], str) or not training[name].strip()
        ):
            raise ConfigError(f"training.{name} must be a non-empty path or omitted")
    if training.get("init_checkpoint") and training.get("resume_checkpoint"):
        raise ConfigError("training.init_checkpoint and training.resume_checkpoint are mutually exclusive")
    if "trainable_patterns" in training:
        patterns = training["trainable_patterns"]
        if not isinstance(patterns, list) or any(not isinstance(item, str) or not item for item in patterns):
            raise ConfigError("training.trainable_patterns must be a list of non-empty strings")
    for name in (
        "persistent_workers",
        "validate_on_first_epoch",
        "validate_on_final_epoch",
    ):
        if name in training and not isinstance(training[name], bool):
            raise ConfigError(f"training.{name} must be boolean")
    loss_weights = training.get("loss_weights", {})
    if not isinstance(loss_weights, Mapping):
        raise ConfigError("training.loss_weights must be a mapping")
    allowed_loss_names = {
        "family",
        "phase",
        "boundary",
        "quality",
        "expert_quality",
        "tracking",
    }
    unknown_loss_names = set(loss_weights) - allowed_loss_names
    if unknown_loss_names:
        raise ConfigError(
            f"training.loss_weights contains unknown names: {sorted(unknown_loss_names)}"
        )
    for name, value in loss_weights.items():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as error:
            raise ConfigError(
                f"training.loss_weights.{name} must be finite and non-negative"
            ) from error
        if isinstance(value, bool) or not math.isfinite(numeric_value) or numeric_value < 0.0:
            raise ConfigError(f"training.loss_weights.{name} must be finite and non-negative")
    return cast(Config, config)


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return validate_config(config)


def validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    config: Config,
    *,
    allow_normalization_mismatch: bool = False,
) -> None:
    """Reject evaluation with a checkpoint built for a different model schema."""

    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must contain a mapping")
    model_name = checkpoint.get("model_name")
    if model_name not in {"tcn", "gru"}:
        raise ValueError("Checkpoint is missing a supported model_name (tcn or gru)")
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, Mapping):
        raise ValueError("Checkpoint is missing the effective training config")

    mismatches: list[str] = []

    def compare(path: str, saved: Any, current: Any) -> None:
        if saved != current:
            mismatches.append(f"{path}: checkpoint={saved!r}, requested={current!r}")

    saved_project = saved_config.get("project", {})
    saved_features = saved_config.get("features", {})
    saved_model = saved_config.get("model", {})
    if not isinstance(saved_project, Mapping) or not isinstance(saved_features, Mapping):
        raise ValueError("Checkpoint config has invalid project/features sections")
    if not isinstance(saved_model, Mapping):
        raise ValueError("Checkpoint config has an invalid model section")

    compare(
        "project.feature_schema_version",
        saved_project.get("feature_schema_version"),
        config["project"].get("feature_schema_version"),
    )
    # Legacy corrected-v1 checkpoints predate explicit normalization/decoder
    # fields. Compare these fields whenever both sides declare them; a new
    # checkpoint cannot silently claim compatibility with two declared values.
    saved_normalization = saved_config.get("data", {}).get("normalization_version")
    current_normalization = config["data"].get("normalization_version")
    if (
        saved_normalization is not None
        and current_normalization is not None
        and not allow_normalization_mismatch
    ):
        compare("data.normalization_version", saved_normalization, current_normalization)
    saved_decoder = saved_project.get("decoder_version")
    current_decoder = config["project"].get("decoder_version")
    if saved_decoder is not None and current_decoder is not None:
        compare("project.decoder_version", saved_decoder, current_decoder)
    for name in ("input_dim", "continuous_dim", "include_diagnostics"):
        compare(f"features.{name}", saved_features.get(name), config["features"].get(name))
    for name in (
        "family_classes",
        "phase_classes",
        "quality_outputs",
        "pooling",
    ):
        compare(f"model.{name}", saved_model.get(name), config["model"].get(name))
    if "expert_quality_classes" in saved_model or "expert_quality_classes" in config["model"]:
        compare(
            "model.expert_quality_classes",
            saved_model.get("expert_quality_classes", 0),
            config["model"].get("expert_quality_classes", 0),
        )

    saved_architecture = saved_model.get(model_name)
    current_architecture = config["model"].get(model_name)
    if not isinstance(saved_architecture, Mapping) or not isinstance(current_architecture, Mapping):
        raise ValueError(f"Checkpoint/config is missing model.{model_name} architecture")
    architecture_keys = (
        ("channels", "kernel_size", "dilations", "dropout")
        if model_name == "tcn"
        else ("hidden_size", "num_layers")
    )
    for name in architecture_keys:
        compare(
            f"model.{model_name}.{name}",
            saved_architecture.get(name),
            current_architecture.get(name),
        )
    if mismatches:
        raise ValueError(
            "Checkpoint is incompatible with the requested config: " + "; ".join(mismatches)
        )


def resolve_project_root(config_path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve the project independently of the caller's current directory."""

    if project_root is not None:
        return Path(project_root).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    candidates = (config_path.parent, *config_path.parents)
    for candidate in candidates:
        if (candidate / "training").is_dir() and (candidate / "training" / "configs").is_dir():
            return candidate
    return config_path.parent


def configured_source_path(config: Config, source: SourceConfig, project_root: Path) -> Path:
    return project_root / source["path"]
