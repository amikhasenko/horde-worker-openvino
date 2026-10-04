"""Minimal OpenVINO CLIP ViT-L/14 image/text similarity, for the CSAM heuristic.

Only the image-embedding path runs per job; text embeddings of the ~22-word CSAM
vocabulary are cached, in the same ``clip_cache/embeds/ViT-L_14/text`` layout the
Alchemist worker uses, so the two share them.

Defaults to the **CPU** so the iGPU stays free for image generation (CLIP on the GPU
would cost ~2.4 GB more shared memory).

Numerically matches clipfree's ``Interrogator.similarity``: L2-normalised embeddings,
raw dot product, rounded to 4 dp.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np
import openvino as ov
from PIL import Image

from clip_tokenizer import tokenize as clip_tokenize

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _clip_image(img: Image.Image, size: int = 224) -> np.ndarray:
    """openai CLIP preprocess: convert to RGB, shortest edge -> size, centre crop.

    (openai's transform converts *after* the resize, unlike the safety checker.)
    """
    img = img.convert("RGB") if img.mode != "RGB" else img
    w, h = img.size
    if w <= h:
        new_w, new_h = size, int(size * h / w)
    else:
        new_h, new_w = size, int(size * w / h)
    img = img.resize((new_w, new_h), Image.BICUBIC)
    w, h = img.size
    i, j = int(round((h - size) / 2.0)), int(round((w - size) / 2.0))
    img = img.crop((j, i, j + size, i + size))
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    arr = arr.transpose(2, 0, 1)  # CHW
    arr = (arr - MEAN[:, None, None]) / STD[:, None, None]
    return np.ascontiguousarray(arr, dtype=np.float32)


def _run(cm, inputs):
    res = cm(inputs)
    return np.asarray(res[cm.output(0)])


class OVClip:
    def __init__(self, ov_dir: str, cache_home: str, device: str = "CPU"):
        core = ov.Core()
        cfg = {"INFERENCE_PRECISION_HINT": "f32"}
        self.visual = core.compile_model(f"{ov_dir}/clip_visual.xml", device, cfg)
        self.text = core.compile_model(f"{ov_dir}/clip_text.xml", device, cfg)
        self.bpe = os.path.join(ov_dir, "bpe_simple_vocab_16e6.txt.gz")
        self._cache = os.path.join(cache_home, "clip_cache", "embeds", "ViT-L_14", "text")
        os.makedirs(self._cache, exist_ok=True)
        self._text_cache: dict[str, np.ndarray] = {}

    def _text_embed(self, text: str) -> np.ndarray:
        if text in self._text_cache:
            return self._text_cache[text]
        path = os.path.join(self._cache, hashlib.sha256(text.encode()).hexdigest() + ".npy")
        if os.path.exists(path):
            emb = np.load(path).astype(np.float32)
        else:
            tokens = clip_tokenize(text, context_length=77, truncate=True, bpe_path=self.bpe)
            emb = _run(self.text, [tokens]).astype(np.float32)
            emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
            np.save(path, emb)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        self._text_cache[text] = emb
        return emb

    def similarity(self, image: Image.Image, texts: list[str]) -> dict[str, float]:
        """{text: cosine similarity rounded to 4 dp}, sorted descending (as clipfree)."""
        feats = _run(self.visual, [_clip_image(image)[None, ...]]).astype(np.float32)
        feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
        out = {}
        for t in texts:
            out[t] = float(round(float((self._text_embed(t) @ feats.T)[0][0]), 4))
        return dict(sorted(out.items(), key=lambda kv: kv[1], reverse=True))
