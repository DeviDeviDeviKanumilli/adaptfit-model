"""Model factory and parameter accounting."""

from __future__ import annotations

from torch import nn

from ..config import Config
from .gru import CausalGRU
from .tcn import CausalTCN


def build_model(name: str, config: Config) -> nn.Module:
    model_config = config["model"]
    common = {
        "input_dim": int(config["features"]["input_dim"]),
        "family_classes": int(model_config["family_classes"]),
        "phase_classes": int(model_config["phase_classes"]),
        "quality_outputs": int(model_config["quality_outputs"]),
        "expert_quality_classes": int(model_config.get("expert_quality_classes", 0)),
        "pooling": str(model_config.get("pooling", "last_valid")),
    }
    if name == "tcn":
        tcn = model_config["tcn"]
        return CausalTCN(
            **common,
            channels=int(tcn["channels"]),
            kernel_size=int(tcn["kernel_size"]),
            dilations=tuple(int(value) for value in tcn["dilations"]),
            dropout=float(tcn["dropout"]),
        )
    if name == "gru":
        gru = model_config["gru"]
        return CausalGRU(
            **common,
            hidden_size=int(gru["hidden_size"]),
            num_layers=int(gru["num_layers"]),
        )
    raise ValueError(f"Unknown model {name!r}; expected tcn or gru")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
