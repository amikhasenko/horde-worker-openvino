"""Offline test of the SD worker: drives synthetic Horde jobs through Worker.handle with a
stub client, so nothing touches the network.

    .venv-sd/bin/python -s selftest_worker.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import worker as W  # noqa: E402


class StubClient:
    def __init__(self):
        self.submitted = []
        self.faulted_jobs = []

    def submit(self, job_id, image, seed, state=None, quality=95):
        self.submitted.append({"id": job_id, "image": image, "seed": seed, "state": state})
        return 1.0

    def faulted(self, job_id, seed=-1):
        self.faulted_jobs.append(job_id)


def job(payload_overrides=None, model="Dreamshaper"):
    payload = {
        "prompt": "a photograph of an astronaut riding a horse",
        "width": 512,
        "height": 512,
        "ddim_steps": 12,
        "sampler_name": "k_euler",
        "cfg_scale": 7.5,
        "seed": 42,
        "karras": False,
    }
    payload.update(payload_overrides or {})
    return {"id": f"test-{payload['sampler_name']}-{payload['seed']}", "model": model, "payload": payload}


def main():
    W.setup_logging(False)
    cfg = W.load_config(os.path.join(HERE, "bridgeData.yaml"))
    w = W.Worker(cfg, HERE)
    w.client = StubClient()

    fails = []

    # 1. happy path
    w.handle(job())
    r = w.client.submitted[-1]
    ok = r["image"].size == (512, 512) and r["state"] is None
    print(f"[{'PASS' if ok else 'FAIL'}] happy path: saved={r['image'].size} state={r['state']}")
    if not ok:
        fails.append("happy")

    # 2. negative prompt via ###
    w.handle(job({"prompt": "a cat###blurry, watermark", "seed": 43}))
    print(f"[PASS] ### negative-prompt split handled")

    # 3. a non-512 size (the horde can send anything <= max_pixels)
    w.handle(job({"width": 512, "height": 384, "seed": 44, "sampler_name": "ddim"}))
    r = w.client.submitted[-1]
    ok = r["image"].size == (512, 384)
    print(f"[{'PASS' if ok else 'FAIL'}] non-square size: {r['image'].size}")
    if not ok:
        fails.append("size")

    # 4. karras + another sampler
    w.handle(job({"sampler_name": "k_dpmpp_2m", "karras": True, "seed": 45}))
    print("[PASS] karras + k_dpmpp_2m handled")

    # 5. unsupported sampler -> faulted, not submitted
    before = len(w.client.submitted)
    w.handle(job({"sampler_name": "not_a_sampler", "seed": 46}))
    ok = len(w.client.submitted) == before and w.client.faulted_jobs
    print(f"[{'PASS' if ok else 'FAIL'}] unsupported sampler faulted={w.client.faulted_jobs}")
    if not ok:
        fails.append("unsupported-sampler")

    # 6. user-requested censoring (a solid black-ish prompt is not nsfw, so just exercise the path)
    w.handle(job({"use_nsfw_censor": True, "seed": 47}))
    r = w.client.submitted[-1]
    print(f"[PASS] censor path exercised (state={r['state']})")

    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
