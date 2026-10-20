"""Model registry. Every model maps (screenshot, instruction) -> (raw text, point in original pixels)."""

from __future__ import annotations

from .base import CenterBaseline, GroundingModel, RandomBaseline


def load_model(name: str, **kwargs) -> GroundingModel:
    if name == "center":
        return CenterBaseline()
    if name == "random":
        return RandomBaseline(seed=kwargs.get("seed", 0))
    if name.startswith("qwen"):
        from .qwen_vl import QwenVL  # imported lazily so tests don't need torch

        return QwenVL(**kwargs)
    raise ValueError(f"unknown model {name!r}")
