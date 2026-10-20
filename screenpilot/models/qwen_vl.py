"""Zero-shot (and later, fine-tuned) inference for Qwen-VL style models via transformers.

The one subtle part is the coordinate space. Qwen2.5-VL resizes each image to a
multiple of 28 px and answers in pixels of that resized image, so we map back
to the original size. Other models answer on a 0-1000 grid or in original
pixels; set coord_space accordingly.
"""

from __future__ import annotations

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from ..parse import parse_point
from .base import GroundingModel

PROMPT = (
    "You are looking at a screenshot. Find the UI element described by the instruction "
    "and give the point to click.\n"
    "Instruction: {instruction}\n"
    'Answer with JSON only, like {{"point_2d": [x, y]}}.'
)


class QwenVL(GroundingModel):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        coord_space: str = "resized",  # resized | normalized_1000 | original
        max_pixels: int = 1280 * 28 * 28,
        max_new_tokens: int = 64,
        adapter: str | None = None,  # path to a LoRA adapter after week 3
        **_,
    ):
        self.name = model_id.split("/")[-1] + (f"+{adapter}" if adapter else "")
        self.coord_space = coord_space
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(model_id, max_pixels=max_pixels)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image, instruction: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT.format(instruction=instruction)},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        raw = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
        point = parse_point(raw)
        if point is None:
            return raw, None
        return raw, self._to_original(point, image.size, inputs)

    def _to_original(self, point, orig_size, inputs):
        x, y = point
        W, H = orig_size
        if self.coord_space == "original":
            return x, y
        if self.coord_space == "normalized_1000":
            return x / 1000 * W, y / 1000 * H
        # resized: image_grid_thw is (t, h/14, w/14) for the resized image
        _, gh, gw = inputs["image_grid_thw"][0].tolist()
        rh, rw = gh * 14, gw * 14
        # Multiply by ratio to convert values back to original coord space.
        return x * W / rw, y * H / rh
