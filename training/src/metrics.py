"""Evaluation metrics for phase, repetitions, family, and quality outputs."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

from .data.schema import PHASE_NAMES


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else float("nan")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values.astype(np.float64) - np.max(values, axis=-1, keepdims=True)
    exponential = np.exp(np.clip(shifted, -60.0, 60.0))
    return exponential / np.maximum(exponential.sum(axis=-1, keepdims=True), 1e-12)


def _confidence_summary(
    logits: np.ndarray,
    valid: np.ndarray,
    prefix: str,
    abstention_threshold: float,
) -> dict[str, float]:
    if not valid.any():
        return {
            f"{prefix}_mean_confidence": float("nan"),
            f"{prefix}_abstention_rate": float("nan"),
        }
    probabilities = _softmax(logits[valid])
    confidence = probabilities.max(axis=-1)
    return {
        f"{prefix}_mean_confidence": float(confidence.mean()),
        f"{prefix}_abstention_rate": float((confidence < abstention_threshold).mean()),
    }


def _macro_f1(target: np.ndarray, prediction: np.ndarray, labels: Iterable[int] | None = None) -> float:
    if target.size == 0:
        return float("nan")
    return float(f1_score(target, prediction, labels=list(labels) if labels is not None else None, average="macro", zero_division=0))


def _binary_f1(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if target.size == 0:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    precision, recall, f1, _ = precision_recall_fscore_support(
        target, prediction, average="binary", zero_division=0
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def expected_calibration_error(probability: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    if probability.size == 0:
        return float("nan")
    if bins <= 0:
        raise ValueError("bins must be positive")
    probability = np.clip(probability.astype(np.float64), 0.0, 1.0)
    correct = correct.astype(np.float64)
    total = len(probability)
    error = 0.0
    for lower, upper in zip(np.linspace(0.0, 1.0, bins, endpoint=False), np.linspace(0.0, 1.0, bins + 1)[1:]):
        selected = (probability >= lower) & ((probability < upper) if upper < 1.0 else (probability <= upper))
        if selected.any():
            error += selected.mean() * abs(probability[selected].mean() - correct[selected].mean())
    return float(error)


def _validate_metric_contract(
    outputs: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> None:
    required_outputs = ("family_logits", "phase_logits", "boundary_logits", "quality_logits")
    required_targets = ("family", "phase", "boundary", "quality", "quality_mask", "time_mask")
    missing_outputs = [key for key in required_outputs if key not in outputs]
    missing_targets = [key for key in required_targets if key not in targets]
    if missing_outputs or missing_targets:
        raise ValueError(
            f"Metric computation is missing outputs={missing_outputs} targets={missing_targets}"
        )

    family_logits = np.asarray(outputs["family_logits"])
    phase_logits = np.asarray(outputs["phase_logits"])
    boundary_logits = np.asarray(outputs["boundary_logits"])
    quality_logits = np.asarray(outputs["quality_logits"])
    family_target = np.asarray(targets["family"])
    phase_target = np.asarray(targets["phase"])
    boundary_target = np.asarray(targets["boundary"])
    quality_target = np.asarray(targets["quality"])
    quality_mask = np.asarray(targets["quality_mask"])
    time_mask = np.asarray(targets["time_mask"])

    if family_logits.ndim != 2:
        raise ValueError("family_logits must have shape [samples, classes]")
    samples = family_logits.shape[0]
    if phase_logits.ndim != 3 or phase_logits.shape[:2] != (samples, time_mask.shape[1] if time_mask.ndim == 2 else -1):
        raise ValueError("phase_logits must have shape [samples, frames, classes]")
    if time_mask.ndim != 2 or phase_logits.shape[:2] != time_mask.shape:
        raise ValueError("phase_logits and time_mask must share [samples, frames]")
    if boundary_logits.ndim != 3 or boundary_logits.shape[:2] != time_mask.shape or boundary_logits.shape[-1] != 2:
        raise ValueError("boundary_logits must have shape [samples, frames, 2]")
    if quality_logits.ndim != 2 or quality_logits.shape != (samples, 4):
        raise ValueError("quality_logits must have shape [samples, 4]")
    if family_target.shape != (samples,):
        raise ValueError("family targets must have shape [samples]")
    if phase_target.shape != time_mask.shape:
        raise ValueError("phase targets and time_mask must share [samples, frames]")
    if boundary_target.shape != (samples, time_mask.shape[1], 2):
        raise ValueError("boundary targets must have shape [samples, frames, 2]")
    if quality_target.shape != (samples, 4) or quality_mask.shape != (samples, 4):
        raise ValueError("quality targets and masks must have shape [samples, 4]")

    for name, values in (
        ("family_logits", family_logits),
        ("phase_logits", phase_logits),
        ("boundary_logits", boundary_logits),
        ("quality_logits", quality_logits),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{name} must be finite")
    if np.any((family_target < -1) | (family_target >= family_logits.shape[-1])):
        raise ValueError("family targets must be -1 or within the family class range")
    if np.any((phase_target < -1) | (phase_target >= phase_logits.shape[-1])):
        raise ValueError("phase targets must be -1 or within the phase class range")
    if not np.isin(boundary_target, (-1.0, 0.0, 1.0)).all():
        raise ValueError("boundary targets must be -1, 0, or 1")
    if np.any(quality_mask & ~np.isfinite(quality_target)):
        raise ValueError("masked quality targets must be finite")
    if np.any(np.isfinite(quality_target) & ((quality_target < 0.0) | (quality_target > 1.0))):
        raise ValueError("quality targets must be within [0, 1]")

    # The expert score is optional and deliberately separate from the four
    # dimension-specific quality outputs. Legacy models may receive default
    # masked arrays without having an expert-quality head.
    if "expert_quality_logits" in outputs:
        if "expert_quality" not in targets or "expert_quality_mask" not in targets:
            raise ValueError(
                "expert_quality_logits requires expert_quality and expert_quality_mask targets"
            )
        expert_logits = np.asarray(outputs["expert_quality_logits"])
        expert_target = np.asarray(targets["expert_quality"])
        expert_mask = np.asarray(targets["expert_quality_mask"]).astype(bool)
        if expert_logits.ndim != 2 or expert_logits.shape[0] != samples or expert_logits.shape[1] < 2:
            raise ValueError("expert_quality_logits must have shape [samples, classes]")
        if expert_target.shape != (samples,) or expert_mask.shape != (samples,):
            raise ValueError("expert quality targets and masks must have shape [samples]")
        if not np.isfinite(expert_logits).all():
            raise ValueError("expert_quality_logits must be finite")
        if np.any((expert_target < -1) | (expert_target >= expert_logits.shape[-1])):
            raise ValueError("expert quality targets must be -1 or within the expert class range")
        if np.any(expert_mask & (expert_target < 0)):
            raise ValueError("expert quality masks cannot be true for unavailable targets")
        if np.any((~expert_mask) & (expert_target != -1)):
            raise ValueError("unmasked expert quality targets must be -1")

    if "tracking_logits" in outputs or "tracking_target" in targets:
        if "tracking_logits" not in outputs or "tracking_target" not in targets:
            raise ValueError("tracking_logits and tracking_target must be provided together")
        tracking_logits = np.asarray(outputs["tracking_logits"])
        tracking_target = np.asarray(targets["tracking_target"])
        if tracking_logits.shape != time_mask.shape or tracking_target.shape != time_mask.shape:
            raise ValueError("tracking logits and targets must match time_mask")
        if not np.isfinite(tracking_logits).all():
            raise ValueError("tracking_logits must be finite")
        if np.any((tracking_target < -1.0) | (tracking_target > 1.0)):
            raise ValueError("tracking targets must be -1 or within [0, 1]")


def compute_metrics(
    outputs: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
    abstention_threshold: float = 0.6,
) -> dict[str, float]:
    if not 0.0 < abstention_threshold < 1.0:
        raise ValueError("abstention_threshold must be between 0 and 1")
    _validate_metric_contract(outputs, targets)
    family_logits = outputs["family_logits"]
    phase_logits = outputs["phase_logits"]
    boundary_logits = outputs["boundary_logits"]
    quality_logits = outputs["quality_logits"]
    family_target = targets["family"]
    phase_target = targets["phase"]
    boundary_target = targets["boundary"]
    quality_target = targets["quality"]
    quality_mask = targets["quality_mask"].astype(bool)
    time_mask = targets["time_mask"].astype(bool)

    family_prediction = family_logits.argmax(axis=-1)
    family_valid = family_target >= 0
    phase_prediction = phase_logits.argmax(axis=-1)
    phase_valid = (phase_target >= 0) & time_mask
    boundary_prediction = (_sigmoid(boundary_logits) >= 0.5).astype(np.int64)
    boundary_valid = (boundary_target >= 0) & time_mask[..., None]

    metrics: dict[str, float] = {
        "family_macro_f1": _macro_f1(family_target[family_valid], family_prediction[family_valid]),
        "family_accuracy": (
            float(np.mean(family_prediction[family_valid] == family_target[family_valid]))
            if family_valid.any()
            else float("nan")
        ),
        "phase_macro_f1": _macro_f1(phase_target[phase_valid], phase_prediction[phase_valid]),
        "phase_frame_accuracy": (
            float(np.mean(phase_prediction[phase_valid] == phase_target[phase_valid]))
            if phase_valid.any()
            else float("nan")
        ),
        "phase_labeled_frame_count": float(phase_valid.sum()),
    }
    phase_f1, _, _, phase_support = precision_recall_fscore_support(
        phase_target[phase_valid],
        phase_prediction[phase_valid],
        labels=list(range(len(PHASE_NAMES))),
        zero_division=0,
    ) if phase_valid.any() else (
        np.full(len(PHASE_NAMES), np.nan, dtype=np.float64),
        np.full(len(PHASE_NAMES), np.nan, dtype=np.float64),
        np.full(len(PHASE_NAMES), np.nan, dtype=np.float64),
        np.zeros(len(PHASE_NAMES), dtype=np.int64),
    )
    for name, score, support in zip(PHASE_NAMES, phase_f1, phase_support):
        metrics[f"phase_f1_{name}"] = float(score) if int(support) > 0 else float("nan")
        metrics[f"phase_support_{name}"] = float(support)
    metrics.update(_confidence_summary(family_logits, family_valid, "family", abstention_threshold))
    metrics.update(_confidence_summary(phase_logits, phase_valid, "phase", abstention_threshold))
    boundary_scores = []
    for index, name in enumerate(("rep_start", "rep_end")):
        result = _binary_f1(boundary_target[..., index][boundary_valid[..., index]].astype(np.int64), boundary_prediction[..., index][boundary_valid[..., index]])
        for key, value in result.items():
            metrics[f"{name}_{key}"] = value
        boundary_scores.append(result["f1"])
    metrics["boundary_f1"] = _finite_mean(boundary_scores)

    quality_prediction = (_sigmoid(quality_logits) >= 0.5).astype(np.int64)
    quality_scores = []
    quality_coverage = []
    for index, name in enumerate(("rom", "tempo", "smoothness", "trunk")):
        valid = quality_mask[:, index] & np.isfinite(quality_target[:, index])
        result = _binary_f1(quality_target[:, index][valid].astype(np.int64), quality_prediction[:, index][valid])
        metrics[f"quality_{name}_f1"] = result["f1"]
        quality_scores.append(result["f1"])
        quality_coverage.append(float(valid.mean()) if len(valid) else 0.0)
    metrics["quality_macro_f1"] = _finite_mean(quality_scores)
    metrics["quality_label_coverage"] = float(np.mean(quality_coverage)) if quality_coverage else 0.0

    if "expert_quality_logits" in outputs:
        expert_logits = np.asarray(outputs["expert_quality_logits"])
        expert_target = np.asarray(targets["expert_quality"])
        expert_mask = np.asarray(targets["expert_quality_mask"]).astype(bool)
        expert_valid = expert_mask & (expert_target >= 0)
        expert_prediction = expert_logits.argmax(axis=-1)
        metrics["expert_quality_accuracy"] = (
            float(np.mean(expert_prediction[expert_valid] == expert_target[expert_valid]))
            if expert_valid.any()
            else float("nan")
        )
        metrics["expert_quality_macro_f1"] = (
            _macro_f1(
                expert_target[expert_valid],
                expert_prediction[expert_valid],
                labels=range(expert_logits.shape[-1]),
            )
            if expert_valid.any()
            else float("nan")
        )
        metrics["expert_quality_mae"] = (
            float(np.mean(np.abs(expert_prediction[expert_valid] - expert_target[expert_valid])))
            if expert_valid.any()
            else float("nan")
        )
        metrics["expert_quality_support"] = float(expert_valid.sum())
        metrics["expert_quality_label_coverage"] = float(expert_valid.mean())

    true_rep_count = ((boundary_target[..., 0] > 0.5) & boundary_valid[..., 0]).sum(axis=1)
    predicted_rep_count = ((boundary_prediction[..., 0] > 0) & time_mask).sum(axis=1)
    metrics["rep_count_mae"] = float(np.mean(np.abs(true_rep_count - predicted_rep_count))) if len(true_rep_count) else float("nan")

    family_probability = _softmax(family_logits)
    family_confidence = family_probability.max(axis=-1)
    family_correct = (family_prediction == family_target).astype(np.float32)
    metrics["family_ece"] = expected_calibration_error(family_confidence[family_valid], family_correct[family_valid])

    # Tracking confidence is an observability signal derived from landmark
    # visibility and pose confidence. It is deliberately not treated as a
    # movement-quality or clinical uncertainty label.
    tracking_logits = outputs.get("tracking_logits")
    tracking_target = targets.get("tracking_target")
    if tracking_logits is not None and tracking_target is not None:
        tracking_logits = np.asarray(tracking_logits)
        tracking_target = np.asarray(tracking_target)
        if tracking_logits.shape != tracking_target.shape or tracking_target.shape != time_mask.shape:
            raise ValueError(
                "tracking_logits, tracking_target, and time_mask must have the same shape"
            )
        tracking_valid = (
            time_mask
            & np.isfinite(tracking_target)
            & (tracking_target >= 0.0)
            & (tracking_target <= 1.0)
        )
        if tracking_valid.any():
            tracking_probability = _sigmoid(tracking_logits)
            metrics["tracking_confidence_mae"] = float(
                np.mean(np.abs(tracking_probability[tracking_valid] - tracking_target[tracking_valid]))
            )
            metrics["tracking_confidence_mean"] = float(tracking_probability[tracking_valid].mean())
            metrics["tracking_target_mean"] = float(tracking_target[tracking_valid].mean())
            metrics["tracking_confidence_abstention_rate"] = float(
                (tracking_probability[tracking_valid] < abstention_threshold).mean()
            )
            valid_frames = int(time_mask.sum())
            metrics["tracking_target_coverage"] = float(
                tracking_valid.sum() / max(valid_frames, 1)
            )
        else:
            metrics["tracking_confidence_mae"] = float("nan")
            metrics["tracking_confidence_mean"] = float("nan")
            metrics["tracking_target_mean"] = float("nan")
            metrics["tracking_confidence_abstention_rate"] = float("nan")
            metrics["tracking_target_coverage"] = 0.0
    metrics["samples"] = float(len(family_target))
    return metrics


def composite_score(metrics: dict[str, float]) -> float:
    values = []
    quality_key = (
        "expert_quality_macro_f1"
        if np.isfinite(metrics.get("expert_quality_macro_f1", float("nan")))
        else "quality_macro_f1"
    )
    for key, weight in (
        ("family_macro_f1", 0.25),
        ("phase_macro_f1", 0.30),
        ("boundary_f1", 0.30),
        (quality_key, 0.15),
    ):
        value = metrics.get(key, float("nan"))
        if np.isfinite(value):
            values.append((value, weight))
    if not values:
        return float("-inf")
    return float(sum(value * weight for value, weight in values) / sum(weight for _, weight in values))
