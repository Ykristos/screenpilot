from __future__ import annotations

import random
from abc import ABC, abstractmethod

from PIL import Image


class GroundingModel(ABC):
    name = "base"

    @abstractmethod
    def predict(self, image: Image.Image, instruction: str) -> tuple[str, tuple[float, float] | None]:
        """Return (raw model text, click point in original image pixels or None)."""


class CenterBaseline(GroundingModel):
    """Always clicks the middle of the screen.

    Two uses: it checks the harness end to end without a GPU, and it is the
    first reward hack an RL policy finds.
    """

    name = "center"

    def predict(self, image, instruction):
        w, h = image.size
        return "center", (w / 2, h / 2)


class RandomBaseline(GroundingModel):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def predict(self, image, instruction):
        w, h = image.size
        return "random", (self.rng.uniform(0, w), self.rng.uniform(0, h))
