"""Standalone iGPU benchmark of SD1.5 via OpenVINO — no Horde, no worker.

Measures what decides whether an OpenVINO SD worker is worth building: seconds per
UNet step, VAE/text-encoder overhead, and the resulting MPS (megapixelsteps/second —
the unit the Horde scores kudos by).

    .venv-sd/bin/python -s bench_sd_ov.py [--steps 20] [--size 512] [--runs 3]

Per-step cost is derived as (t(N steps) - t(1 step)) / (N-1), which cancels out the
fixed text-encoder + VAE-decode overhead; the remainder is reported separately.
"""
from __future__ import annotations

import argparse
import statistics
import time

PROMPT = "a photograph of an astronaut riding a horse"
NEGATIVE = "blurry, low quality, watermark, text"


def timed(pipe, steps, size, seed, extra=None):
    t = time.time()
    out = pipe(
        PROMPT,
        negative_prompt=NEGATIVE,
        num_inference_steps=steps,
        guidance_scale=7.5,
        height=size,
        width=size,
        generator=None,
        output_type="np",
        **(extra or {}),
    )
    return time.time() - t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="an exported SD1.5 OpenVINO pipeline dir")
    ap.add_argument("--device", default="GPU")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args()

    import openvino as ov
    from optimum.intel import OVStableDiffusionPipeline

    core = ov.Core()
    print(f"OpenVINO {ov.__version__} | devices: {core.available_devices}")
    print(f"device requested: {args.device} | dir: {args.dir}")

    t = time.time()
    pipe = OVStableDiffusionPipeline.from_pretrained(args.dir)
    print(f"loaded pipeline in {time.time() - t:.1f}s")

    pipe.to(device=args.device)
    pipe.reshape(batch_size=1, height=args.size, width=args.size, num_images_per_prompt=1)
    t = time.time()
    pipe.compile()
    print(f"compiled in {time.time() - t:.1f}s")

    # Report which device actually ended up in use.
    try:
        dev = pipe.unet.request.get_compiled_model().get_property("DEVICE_NAME")
        print(f"unet running on: {dev}")
    except Exception as e:  # noqa: BLE001
        print(f"(device introspection unavailable: {e})")

    print("warmup...")
    timed(pipe, args.steps, args.size, 0)
    print("warmup done")

    t1 = statistics.median([timed(pipe, 1, args.size, 0)[0] for _ in range(args.runs)])
    tn = statistics.median([timed(pipe, args.steps, args.size, 0)[0] for _ in range(args.runs)])

    per_step = (tn - t1) / (args.steps - 1)
    overhead = t1 - per_step
    mp = (args.size / 1000.0) * (args.size / 1000.0)  # megapixels
    mps = (mp * args.steps) / tn
    fps = 1.0 / tn

    print()
    print(f"=== SD1.5 {args.size}x{args.size}, {args.steps} steps, {args.device} ===")
    print(f"t(1 step)      : {t1:.2f} s   (text enc + 1 unet + vae decode)")
    print(f"t({args.steps} steps)    : {tn:.2f} s")
    print(f"per UNet step  : {per_step:.2f} s")
    print(f"fixed overhead : {overhead:.2f} s   (text encoder + VAE decode)")
    print(f"seconds/image  : {tn:.1f} s   ({fps * 60:.1f} images/hour)")
    print(f"MPS            : {mps:.4f} megapixelsteps/s")
    print(f"  (Horde's 'extra slow worker' guidance is for <0.3 MPS)")


if __name__ == "__main__":
    raise SystemExit(main())
