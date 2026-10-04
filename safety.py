"""NSFW check for generated images, using the OpenVINO safety checker IR
(``<ov_dir>/safety_visual.xml`` + ``safety_meta.npz``, built by
``tools/convert_ov_assets.py``).

Reproduces diffusers' ``StableDiffusionSafetyChecker.forward``: cosine distance to the
concept/special-care embeddings, the 0.01 ``adjustment`` the special-care pass adds, and
the 3-dp rounding before each threshold test.
"""
from __future__ import annotations

import numpy as np
import openvino as ov
from PIL import Image

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _preprocess(img: Image.Image, size: int = 224) -> np.ndarray:
    """transformers CLIPFeatureExtractor: RGB first, then shortest-edge resize + centre crop."""
    img = img.convert("RGB")
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_h, new_w = size, int(size * w / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    w, h = img.size
    left, top = int((w - size) / 2), int((h - size) / 2)
    img = img.crop((left, top, left + size, top + size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)
    arr = (arr - MEAN[:, None, None]) / STD[:, None, None]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _normalize(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


class SafetyChecker:
    def __init__(self, ov_dir: str, device: str = "CPU"):
        self.ov_dir = ov_dir
        core = ov.Core()
        self.model = core.compile_model(f"{ov_dir}/safety_visual.xml", device)
        meta = np.load(f"{ov_dir}/safety_meta.npz")
        self.concept_embeds = _normalize(meta["concept_embeds"].astype(np.float32))
        self.special_embeds = _normalize(meta["special_care_embeds"].astype(np.float32))
        self.concept_weights = meta["concept_embeds_weights"].astype(np.float32)
        self.special_weights = meta["special_care_embeds_weights"].astype(np.float32)

    def is_nsfw(self, image: Image.Image) -> bool:
        chw = _preprocess(image)[None, ...]
        res = self.model([chw])
        embeds = _normalize(np.asarray(res[self.model.output(0)], dtype=np.float32))
        special_cos = embeds @ self.special_embeds.T
        cos = embeds @ self.concept_embeds.T
        for i in range(embeds.shape[0]):
            adjustment = 0.0
            for c in range(special_cos.shape[1]):
                if round(float(special_cos[i][c] - self.special_weights[c] + adjustment), 3) > 0:
                    adjustment = 0.01
            bad = 0
            for c in range(cos.shape[1]):
                if round(float(cos[i][c] - self.concept_weights[c] + adjustment), 3) > 0:
                    bad += 1
            if bad > 0:
                return True
        return False
