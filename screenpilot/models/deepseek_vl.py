"""Zero-shot (and later, fine-tuned) inference for DeepSeek-VL style models via transformers.

We map back to the original size. This model answers on a 0-999 grid; 
set coord_space accordingly.
"""

from __future__ import annotations

import torch
from PIL import Image
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

from ..parse import parse_point
from .base import GroundingModel


class DeepSeekVL(GroundingModel):
    def __init__(
            self,
            model_id: str = "deepseek-ai/deepseek-vl2-tiny",
            coord_space:str = "normalized_999",
            max_new_tokens: int = 128,
            adapter: str | None = None,
            **_,
    ):
        self.name = model_id.split("/")[-1] + (f"+{adapter}" if adapter else "")
        self.coord_space = coord_space
        self.max_new_tokens = max_new_tokens
        self.processor = DeepseekVLV2Processor.from_pretrained(model_id)
        self.tokenizer = self.processor.tokenizer
        self.model = DeepseekVLV2ForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        self.model.eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image, instruction: str):
        messages = [
            {"role": "<|User|>", "content": "<image>\n<|ref|>{instruction}<|/ref|>."},
            {"role": "<|Assistant|>", "content": ""},
        ]
        inputs = self.processor(conversations=messages, images=[image], force_batchify=True, system_prompt=""
                                ).to(self.model.device)
        # Runs the vision encoder
        embeds = self.model.prepare_inputs_embeds(**inputs)

        out = self.model.language.generate(
            input_embeds=embeds,
            attention_mask=inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        raw = self.tokenizer.decode(out[0].cpu().tolist, skip_special_tokens=False)
        point = parse_point(raw)
        if point is None:
            return raw, None
        return raw, self._to_original(point, image.size, inputs)

    # TODO: Consolidate reusable logic in shared script.
    def _to_original(self, point, orig_size, inputs):
        x, y = point
        W, H = orig_size
        # DeepSeek normalizes coords to 0-999
        if self.coord_space == "normalized_999":
            return x / 999 * W, y / 999 * H
        raise ValueError(f'coord_space: {self.coord_space} isn\'t supported')