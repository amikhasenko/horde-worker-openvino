"""Add another SD1.5 model to the OpenVINO worker.

    .venv-sd/bin/python -s add_model.py --name "Anything v5"
    .venv-sd/bin/python -s add_model.py --name "MeinaMix" --dry-run

Steps, in order:

  1. look the name up in the Horde model reference (names must match exactly — that is
     what the server validates the pop payload against)
  2. download the single-file checkpoint the Horde itself serves
  3. ckpt/safetensors -> diffusers      (from_single_file, repacking legacy .ckpt)
  4. diffusers -> OpenVINO fp16 IR      (optimum-cli, --weight-format fp16)

then deletes the intermediates and prints the `bridgeData.yaml` lines to add.

Paths come from `bridgeData.yaml` (`cache_home`, `ov_sd_dir`, `sd15_config`); the
subprocesses run with the interpreter this script was started with, so run it from the
same venv as the worker. The Horde model reference is downloaded to
`<cache_home>/horde_model_reference/legacy/stable_diffusion.json` on first use.

**SD1.5 only.** The worker hardcodes SD1.5 semantics (CLIP text encoder, 77-token
context, 8x VAE downscale, 0.18215 scaling) — SDXL/SD2.1/Flux are a different pipeline
and would need real work, not a config line. The script refuses a non-SD1 baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REFERENCE_URL = (
    "https://raw.githubusercontent.com/Haidra-Org/AI-Horde-image-model-reference/"
    "main/stable_diffusion.json"
)
PYTHON = sys.executable
OPTIMUM = os.path.join(os.path.dirname(PYTHON), "optimum-cli")

# Paths, filled in by main() from bridgeData.yaml.
CACHE = ""
OV_SD = ""
CONFIG_DIFFUSERS = ""
REFERENCE = ""


def load_paths(config_path: str) -> None:
    global CACHE, OV_SD, CONFIG_DIFFUSERS, REFERENCE
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not cfg.get("cache_home"):
        raise SystemExit(f"{config_path} has no cache_home; copy bridgeData_template.yaml first")
    CACHE = cfg["cache_home"]
    OV_SD = cfg.get("ov_sd_dir") or os.path.join(CACHE, "ov_sd")
    CONFIG_DIFFUSERS = cfg.get("sd15_config") or "stable-diffusion-v1-5/stable-diffusion-v1-5"
    REFERENCE = os.path.join(CACHE, "horde_model_reference", "legacy", "stable_diffusion.json")


def ensure_reference() -> None:
    """Fetch the Horde's model reference (it is not shipped with this repo)."""
    if os.path.exists(REFERENCE) and os.path.getsize(REFERENCE) > 0:
        return
    os.makedirs(os.path.dirname(REFERENCE), exist_ok=True)
    print(f"[0/4] downloading the Horde model reference -> {REFERENCE}")
    urllib.request.urlretrieve(REFERENCE_URL, REFERENCE)


def lookup(name: str) -> dict:
    ref = json.loads(open(REFERENCE, encoding="utf-8").read())
    if name not in ref:
        # suggest near misses (typos included) rather than dumping 163 names
        import difflib

        near = difflib.get_close_matches(name, list(ref), n=6, cutoff=0.5)
        if not near:
            near = [k for k in ref if name.lower() in k.lower()][:6]
        raise SystemExit(f"model {name!r} not in the Horde reference. Did you mean: {near}")
    e = ref[name]
    baseline = str(e.get("baseline", "")).lower()
    if baseline != "stable diffusion 1":
        raise SystemExit(f"{name!r} has baseline {e.get('baseline')!r}; only 'stable diffusion 1' (SD1.5) is supported")
    if e.get("inpainting"):
        print(f"WARNING: {name!r} is an inpainting model — the worker only does text-to-image.")
    dl = (e.get("config") or {}).get("download") or []
    if not dl:
        raise SystemExit(f"{name!r} has no download entry in the reference")
    return {
        "name": name,
        "size_gb": round((e.get("size_on_disk_bytes") or 0) / 1e9, 2),
        "url": dl[0].get("file_url"),
        "file": dl[0].get("file_name"),
        "nsfw": e.get("nsfw"),
        "style": e.get("style"),
        "tags": e.get("tags"),
    }


def download(info: dict) -> str:
    raw_dir = os.path.join(CACHE, "sd_raw")
    os.makedirs(raw_dir, exist_ok=True)
    dest = os.path.join(raw_dir, info["file"])
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[1/4] already downloaded: {dest}")
        return dest
    print(f"[1/4] downloading {info['file']} ({info['size_gb']} GB)")
    subprocess.run(["curl", "-L", "--fail", "-sS", "--retry", "3", "-o", dest, info["url"]], check=True)
    return dest


def to_diffusers(raw: str, out: str) -> None:
    print(f"[2/4] {os.path.basename(raw)} -> diffusers ({out})")
    subprocess.run(
        [PYTHON, "-s", os.path.join(HERE, "tools", "convert_sd_single.py"),
         "--ckpt", raw, "--config", CONFIG_DIFFUSERS, "--out", out],
        check=True,
    )


def to_openvino(diffusers_dir: str, out: str) -> None:
    print(f"[3/4] diffusers -> OpenVINO fp16 ({out})")
    env = dict(os.environ)
    env.setdefault("HF_HOME", os.path.join(HERE, ".hf-cache"))
    subprocess.run(
        [OPTIMUM, "export", "openvino", "--model", diffusers_dir,
         "--task", "stable-diffusion", "--weight-format", "fp16", out],
        check=True, env=env,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="exact model name from the Horde reference")
    ap.add_argument("--config", default=os.path.join(HERE, "bridgeData.yaml"),
                    help="bridgeData.yaml to read cache_home/ov_sd_dir/sd15_config from")
    ap.add_argument("--dir-name", default=None, help="output dir (default: name with spaces removed)")
    ap.add_argument("--dry-run", action="store_true", help="show what would happen, do nothing")
    ap.add_argument("--keep-intermediates", action="store_true")
    args = ap.parse_args()

    load_paths(args.config)
    ensure_reference()
    info = lookup(args.name)
    dir_name = args.dir_name or args.name.replace(" ", "").replace("/", "_")
    print(json.dumps(info, indent=1))
    print(f"would export to {OV_SD}/{dir_name}")
    if args.dry_run:
        print("\n[dry-run] nothing downloaded or built")
        return 0

    t = time.time()
    raw = download(info)
    diff = os.path.join(CACHE, "diffusers", dir_name)
    ovdir = os.path.join(OV_SD, dir_name)
    try:
        to_diffusers(raw, diff)
        to_openvino(diff, ovdir)
    finally:
        if not args.keep_intermediates:
            print("[4/4] cleaning intermediates")
            # A legacy .ckpt is repacked to a sibling .safetensors; that file is
            # produced by convert_sd_single.py, not returned here, so derive it.
            repacked = raw[: -len(".ckpt")] + ".safetensors" if raw.endswith(".ckpt") else None
            for p in (raw, repacked, diff):
                # Never let cleanup raise: it runs in a finally and would mask the real error.
                if not p or not os.path.exists(p):
                    continue
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError as e:
                        print(f"    could not remove {p}: {e}")

    print(f"\ndone in {time.time() - t:.0f}s. Add to bridgeData.yaml:\n")
    print("models:")
    print(f'  - "{info["name"]}"')
    print("model_dirs:")
    print(f'  "{info["name"]}": "{dir_name}"')
    print("\nthen restart the worker. Verify with:")
    print(f'  {PYTHON} -s tools/gen_sd_ov.py --dir "{ovdir}" '
          f'--prompt "a test" --out "{os.path.join(HERE, "tmp", dir_name + ".png")}"')
    print(f"\n(disk: ~{2.1:.0f} GB IR + ~2 GB compile cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
