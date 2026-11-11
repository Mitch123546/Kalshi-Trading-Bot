from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .features import FEATURE_NAMES


class ProbabilityModel(Protocol):
    name: str

    def predict(self, features: dict[str, float]) -> float: ...


@dataclass
class DiffusionBaseline:
    name: str = "diffusion-baseline"

    def predict(self, features: dict[str, float]) -> float:
        return float(features["diffusion_probability"])


class TorchMLPModel:
    name = "torch-mlp"

    def __init__(self, artifact_path: Path):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install the ML extras: pip install -e .[ml]") from exc
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if tuple(artifact["feature_names"]) != FEATURE_NAMES:
            raise ValueError("Model feature schema does not match this bot version")
        self.mean = np.asarray(artifact["mean"], dtype=np.float32)
        self.scale = np.asarray(artifact["scale"], dtype=np.float32)
        self.temperature = float(artifact.get("temperature", 1.0))
        self.net = _build_torch_net(len(FEATURE_NAMES), artifact["hidden"])
        self.net.load_state_dict(artifact["state_dict"])
        self.net.eval()
        self.torch = torch

    def predict(self, features: dict[str, float]) -> float:
        values = np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float32)
        values = (values - self.mean) / self.scale
        with self.torch.inference_mode():
            logits = self.net(self.torch.from_numpy(values).unsqueeze(0)).squeeze()
            probability = self.torch.sigmoid(logits / self.temperature).item()
        return float(min(0.999, max(0.001, probability)))


def _build_torch_net(input_size: int, hidden: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_size, hidden),
        nn.GELU(),
        nn.LayerNorm(hidden),
        nn.Dropout(0.15),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(0.10),
        nn.Linear(hidden, 1),
    )


def load_model(path: Path) -> ProbabilityModel:
    return TorchMLPModel(path) if path.exists() else DiffusionBaseline()


def feature_json(features: dict[str, float]) -> str:
    return json.dumps({name: features[name] for name in FEATURE_NAMES}, separators=(",", ":"))


def brier_score(probabilities: list[float], outcomes: list[int]) -> float:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal non-zero lengths")
    return sum((p - y) ** 2 for p, y in zip(probabilities, outcomes, strict=True)) / len(outcomes)


def log_loss(probabilities: list[float], outcomes: list[int]) -> float:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("probabilities and outcomes must have equal non-zero lengths")
    eps = 1e-8
    return -sum(
        y * math.log(max(eps, min(1 - eps, p))) + (1 - y) * math.log(max(eps, min(1 - eps, 1 - p)))
        for p, y in zip(probabilities, outcomes, strict=True)
    ) / len(outcomes)
