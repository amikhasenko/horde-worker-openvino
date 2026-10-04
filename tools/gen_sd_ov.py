"""Generate one image from an exported OpenVINO SD pipeline dir (sanity check).

    python tools/gen_sd_ov.py --dir cache/ov_sd/SomeModel \
        --prompt "a portrait of a knight" --out tmp/out.png
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="blurry, low quality, watermark, text, deformed")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cfg", type=float, default=7.5)
    args = ap.parse_args()

    import torch
    from optimum.intel import OVStableDiffusionPipeline

    pipe = OVStableDiffusionPipeline.from_pretrained(args.dir)
    pipe.to(device=args.device)
    pipe.reshape(batch_size=1, height=args.size, width=args.size, num_images_per_prompt=1)
    pipe.compile()

    gen = torch.Generator().manual_seed(args.seed)
    t = time.time()
    out = pipe(
        args.prompt,
        negative_prompt=args.negative,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        height=args.size,
        width=args.size,
        generator=gen,
        output_type="pil",
    )
    img = out.images[0]
    img.save(args.out)
    print(f"{args.out}  {img.size}  {time.time() - t:.1f}s  prompt={args.prompt!r}")


if __name__ == "__main__":
    raise SystemExit(main())
