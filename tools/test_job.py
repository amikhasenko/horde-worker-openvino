"""Submit one real job aimed at this worker, then poll it to completion.

Useful for checking a freshly configured worker end to end without waiting for the
public queue to hand it something.  The job is pinned to the worker by UUID (the API's
``workers`` field does not accept a name — a name returns HTTP 404 ``WorkerNotFound``)
and marked ``extra_slow_workers``, without which an extra-slow worker is never eligible.

    python tools/test_job.py                          # prompt/model from the defaults
    python tools/test_job.py --prompt "a red apple" --model Deliberate
    python tools/test_job.py --config bridgeData.yaml --width 512 --height 512

Reads the API key from ``HORDE_API_KEY`` or, failing that, from the config file.
Nothing is written outside the current directory except the fetched image.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DEFAULT_PROMPT = "a red apple on a wooden table, still life, warm light, highly detailed"
DEFAULT_NEGATIVE = "blurry, low quality, watermark, text, deformed"


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def find_worker_id(horde_url: str, name: str) -> str | None:
    """Resolve a dreamer name to its worker UUID via GET /api/v2/workers."""
    try:
        r = requests.get(f"{horde_url}/api/v2/workers", timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        print(f"could not list workers: {e}")
        return None
    for w in r.json():
        if w.get("name") == name:
            return w.get("id")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "bridgeData.yaml"))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--negative", default=DEFAULT_NEGATIVE)
    ap.add_argument("--model", default=None, help="default: first model in the config")
    ap.add_argument("--sampler", default="k_euler")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default=str(ROOT / "tmp" / "test_job.webp"))
    ap.add_argument("--timeout", type=int, default=1800, help="seconds to wait for the job")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    horde_url = (cfg.get("horde_url") or "https://aihorde.net").rstrip("/")
    api_key = cfg.get("api_key") or os.environ.get("HORDE_API_KEY", "")
    headers = {"apikey": api_key} if api_key else {}

    name = cfg.get("dreamer_name")
    if not name:
        print(f"no dreamer_name in {args.config}; pass --config or set one")
        return 2
    models = cfg.get("models") or []
    model = args.model or (models[0] if models else None)
    if not model:
        print(f"no model to request; pass --model or add models to {args.config}")
        return 2

    worker_id = find_worker_id(horde_url, name)
    if not worker_id:
        print(f"worker {name!r} is not visible on {horde_url} yet (is it running?)")
        return 1
    print(f"{name} -> {worker_id}")

    # The negative prompt rides on the positive one, '###'-separated (Horde convention).
    prompt = args.prompt + ("###" + args.negative if args.negative else "")
    body = {
        "prompt": prompt,
        "params": {
            "sampler_name": args.sampler,
            "cfg_scale": args.cfg,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "n": 1,
            "seed": str(args.seed),
        },
        "nsfw": True,
        "models": [model],
        "r2": False,
        "extra_slow_workers": True,   # required for an extra-slow worker to be eligible
        "workers": [worker_id],       # pin the job to this worker (UUID)
    }

    print(f"submitting {args.width}x{args.height}/{args.steps} of {model!r}, pinned to {name}")
    r = requests.post(f"{horde_url}/api/v2/generate/async", json=body, headers=headers, timeout=60)
    if not r.ok:
        print(f"async -> HTTP {r.status_code}: {r.text[:400]}")
        return 1
    jid = r.json().get("id")
    if not jid:
        print(f"no job id: {r.text[:400]}")
        return 1
    print(f"job id {jid}")

    status_url = f"{horde_url}/api/v2/generate/status/{jid}"
    deadline = time.monotonic() + args.timeout
    st: dict = {}
    while time.monotonic() < deadline:
        time.sleep(10)
        try:
            st = requests.get(status_url, headers=headers, timeout=60).json()
        except requests.RequestException as e:
            print(f"   status check failed: {e}")
            continue
        if st.get("done"):
            break
        print(f"   waiting... queue_position={st.get('queue_position')} wait={st.get('wait_time')}s")
    else:
        print("timed out waiting for the job")
        return 1

    gens = st.get("generations") or []
    g = gens[0] if gens else {}
    print("\n=== result ===")
    print(f"worker      : {g.get('worker_name')}")
    print(f"model used  : {g.get('model')}")
    print(f"seed        : {g.get('seed')}")
    print(f"state       : {g.get('state')}")
    print(f"kudos       : {g.get('kudos')}")
    if g.get("censored"):
        print("censored    : True")
    if g.get("img") is None:
        print(f"no image in the response (gen_metadata={st.get('gen_metadata')})")
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(base64.b64decode(g["img"]))
    print(f"image       : {out}")
    print("\nserved by our worker:", g.get("worker_name") == name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
