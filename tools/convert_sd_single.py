"""Offline: single-file SD1.5 checkpoint (the format the Horde serves) -> OpenVINO fp16.

The Horde's models are single-file ``.ckpt``/``.safetensors`` (ComfyUI style), not the
diffusers layout `optimum-cli` wants, so this is a two-hop conversion:

    ckpt/safetensors --from_single_file--> diffusers dir --optimum-cli--> OpenVINO IR

Run in a venv with torch + diffusers (this is an offline conversion tool):

    python tools/convert_sd_single.py \
        --ckpt cache/sd_raw/SomeModel.safetensors --out cache/diffusers/SomeModel
    optimum-cli export openvino \
        --model cache/diffusers/SomeModel --task stable-diffusion \
        --weight-format fp16 cache/ov_sd/SomeModel

(tools/add_model.py runs both hops for you, after looking the model up in the Horde
reference.)

fp16 throughout: that is what every Horde generation worker runs (see hordelib's
``half_precision=True`` default), not a degradation.

``--config`` is a diffusers-format SD1.5 directory used as the skeleton for the
conversion — the single-file checkpoints the Horde serves carry no pipeline config.
Either a local path or a Hugging Face repo id (diffusers downloads it).
"""
from __future__ import annotations

import argparse
import os
import shutil
import time

# A diffusers-format SD1.5 pipeline config (scheduler/tokenizer/unet/vae configs).
CONFIG_DIR = os.environ.get("SD15_CONFIG", "stable-diffusion-v1-5/stable-diffusion-v1-5")


def to_safetensors(path: str, workdir: str) -> str:
    """Some Horde ``.ckpt`` files are legacy pickles that torch's ``weights_only=True``
    default refuses.  Re-save them as safetensors so the normal loader can read them."""
    if path.endswith(".safetensors"):
        return path

    import torch
    from safetensors.torch import save_file

    print(f"[convert] {os.path.basename(path)} is a legacy .ckpt; repacking to safetensors")
    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    state = {k: v.contiguous() for k, v in state.items() if isinstance(v, torch.Tensor)}
    print(f"[convert]   {len(state)} tensors")
    out = os.path.join(workdir, os.path.basename(path).rsplit(".", 1)[0] + ".safetensors")
    save_file(state, out)
    del state, obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="single-file .ckpt/.safetensors from the Horde")
    ap.add_argument("--config", default=CONFIG_DIR, help="diffusers SD1.5 config dir")
    ap.add_argument("--out", required=True, help="output diffusers dir")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--clean", action="store_true", help="remove --out first if it exists")
    args = ap.parse_args()

    import torch
    from diffusers import StableDiffusionPipeline

    if args.clean and os.path.isdir(args.out):
        shutil.rmtree(args.out)

    dtype = getattr(torch, args.dtype)
    t = time.time()
    ckpt_path = to_safetensors(args.ckpt, os.path.dirname(os.path.abspath(args.ckpt)))
    print(f"[convert] loading {ckpt_path} (config={args.config}, dtype={args.dtype})")
    pipe = StableDiffusionPipeline.from_single_file(
        ckpt_path,
        config=args.config,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
    )
    print(f"[convert] loaded in {time.time() - t:.1f}s")

    # Drop the (absent) safety checker explicitly so the saved layout is clean.
    pipe.safety_checker = None
    pipe.feature_extractor = None
    pipe.save_pretrained(args.out, safe_serialization=True)
    print(f"[convert] saved diffusers pipeline -> {args.out} ({time.time() - t:.1f}s total)")

    for sub in ("unet", "vae", "text_encoder"):
        p = os.path.join(args.out, sub)
        if os.path.isdir(p):
            print("   ", sub, sorted(os.listdir(p)))


if __name__ == "__main__":
    raise SystemExit(main())
