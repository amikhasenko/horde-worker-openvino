"""Offline converter: CLIP + safety checker -> fp32 OpenVINO IR.

Builds the three IR blobs this worker needs at runtime, under ``--out`` (which is the
``ov_dir`` in your ``bridgeData.yaml``):

  clip_visual.xml    openai CLIP ViT-L/14 image tower  [B,3,224,224] f32 -> [B,768] f32
  clip_text.xml      openai CLIP ViT-L/14 text tower   [B,77] i64      -> [B,768] f32
  bpe_simple_vocab_16e6.txt.gz   (copied from the clip package)
  safety_visual.xml  safety checker vision tower       [B,3,224,224] f32 -> [B,768] f32
  safety_meta.npz    safety checker concept/special-care embeddings + weights

Everything is exported **fp32** (no weight compression, no fp16).  That is not
politeness: CLIP text embeddings are cached as fp32 and the CSAM heuristic compares
similarities, so the exported encoders must reproduce the fp32 reference.  The runtime
additionally forces ``INFERENCE_PRECISION_HINT=f32`` so the GPU plugin does not
silently compute in fp16.

Run this once, in a venv that has torch + openvino (it is an offline conversion tool —
the worker itself never imports torch):

    python tools/convert_ov_assets.py --out /path/to/cache/ov

Both source models are downloaded automatically on first run:

  * CLIP ViT-L/14 via the ``clip`` package (into ``~/.cache/clip``)
  * the safety checker via diffusers (``CompVis/stable-diffusion-safety-checker``)

Derived from AI-Horde-Worker's ``tools/convert_ov.py`` (AGPL-3.0); see NOTICE.
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

# Don't let torch probe a CUDA device we do not have.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np  # noqa: E402
import openvino as ov  # noqa: E402
import torch  # noqa: E402

SAFETY_REPO = "CompVis/stable-diffusion-safety-checker"


def export(module: torch.nn.Module, example_input, out_xml: Path, name: str = "") -> dict:
    """Trace ``module`` and save fp32 IR (xml + bin)."""
    tic = time.time()
    module.eval()
    with torch.no_grad():
        om = ov.convert_model(module, example_input=example_input)
    ov.save_model(om, str(out_xml), compress_to_fp16=False)

    def _out_name(port, i):
        # Models with positional `*args` inputs may not have a stable output name.
        try:
            return port.get_any_name()
        except RuntimeError:
            return f"output_{i}"

    ins = [(p.get_any_name(), str(p.get_partial_shape()), str(p.get_element_type())) for p in om.inputs]
    outs = [(_out_name(p, i), str(p.get_partial_shape())) for i, p in enumerate(om.outputs)]
    print(f"[convert] {name or out_xml.name}: {out_xml.name} in {time.time() - tic:.1f}s")
    for n, s, t in ins:
        print(f"            in  {n}: {s} {t}")
    for n, s in outs:
        print(f"            out {n}: {s}")
    return {"xml": out_xml.name, "inputs": ins, "outputs": outs}


# ----------------------------------------------------------------------------- CLIP
class ClipVisual(torch.nn.Module):
    """openai CLIP ``model.visual``: [B,3,224,224] f32 -> [B,768] f32."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model.visual(image.float())


class ClipText(torch.nn.Module):
    """openai CLIP ``model.encode_text``: [B,77] i64 -> [B,768] f32."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model.encode_text(input_ids)


def convert_clip(out: Path, clip_cache: Path | None) -> dict:
    import clip as openai_clip

    model, _ = openai_clip.load(
        "ViT-L/14", device="cpu", download_root=str(clip_cache) if clip_cache else None
    )
    model = model.float().eval()
    assert not next(model.parameters()).is_cuda, "CLIP must be on CPU"

    info = {
        "clip_visual": export(
            ClipVisual(model), (torch.zeros(1, 3, 224, 224, dtype=torch.float32),),
            out / "clip_visual.xml", name="clip/visual",
        ),
        "clip_text": export(
            ClipText(model), (torch.zeros(1, 77, dtype=torch.int64),),
            out / "clip_text.xml", name="clip/text",
        ),
    }
    # The vendored tokenizer needs the BPE vocabulary that ships with openai clip.
    bpe_src = Path(openai_clip.__file__).parent / "bpe_simple_vocab_16e6.txt.gz"
    shutil.copyfile(bpe_src, out / "bpe_simple_vocab_16e6.txt.gz")
    info["bpe_vocab"] = "bpe_simple_vocab_16e6.txt.gz"
    info["context_length"] = 77
    return info


# ----------------------------------------------------------------------------- safety
class SafetyVisual(torch.nn.Module):
    """diffusers safety checker vision tower: [B,3,224,224] f32 -> projected [B,768] f32."""

    def __init__(self, checker):
        super().__init__()
        self.vision_model = checker.vision_model
        self.visual_projection = checker.visual_projection

    def forward(self, pixel_values):
        pooled = self.vision_model(pixel_values.float())[1]
        return self.visual_projection(pooled)


def convert_safety(out: Path, safety_repo: Path | None) -> dict:
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker

    src = str(safety_repo) if safety_repo else SAFETY_REPO
    checker = StableDiffusionSafetyChecker.from_pretrained(src).eval()

    export(
        SafetyVisual(checker),
        (torch.zeros(1, 3, 224, 224, dtype=torch.float32),),
        out / "safety_visual.xml",
        name="safety/visual",
    )
    np.savez(
        out / "safety_meta.npz",
        concept_embeds=checker.concept_embeds.detach().cpu().float().numpy(),
        special_care_embeds=checker.special_care_embeds.detach().cpu().float().numpy(),
        concept_embeds_weights=checker.concept_embeds_weights.detach().cpu().float().numpy(),
        special_care_embeds_weights=checker.special_care_embeds_weights.detach().cpu().float().numpy(),
    )
    return {
        "safety_visual": "safety_visual.xml",
        "safety_meta": "safety_meta.npz",
        "num_concepts": int(checker.concept_embeds.shape[0]),
        "num_special_care": int(checker.special_care_embeds.shape[0]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dir (your bridgeData.yaml `ov_dir`)")
    ap.add_argument("--only", default="clip,safety", help="comma list of: clip,safety")
    ap.add_argument("--clip-cache", default=None, help="where clip.load() caches ViT-L-14.pt")
    ap.add_argument("--safety-repo", default=None, help=f"local safety checker dir or HF repo id (default {SAFETY_REPO})")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    import json

    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    if "clip" in only:
        manifest.update(convert_clip(out, Path(args.clip_cache) if args.clip_cache else None))
    if "safety" in only:
        manifest.update(convert_safety(out, Path(args.safety_repo) if args.safety_repo else None))

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
