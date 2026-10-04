"""Offline test of the SD worker: drives synthetic Horde jobs through Worker.handle with a
stub client, so nothing touches the network.

    ./run.sh selftest_worker.py        (or: python selftest_worker.py)

Covers the generation paths (samplers, sizes, negative prompts, faulting) and the censor
wiring: which of the three censor reasons fires, and which placeholder it selects.  The
safety classifier is stubbed for the censor tests so they do not depend on generating a
specific NSFW image.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from PIL import Image  # noqa: E402

import worker as W  # noqa: E402

# Filled in by main() from bridgeData.yaml — the model names are configurable.
DEFAULT_MODEL = None


class StubClient:
    def __init__(self):
        self.submitted = []
        self.faulted_jobs = []

    def submit(self, job_id, image, seed, state=None, quality=95):
        self.submitted.append({"id": job_id, "image": image, "seed": seed, "state": state})
        return 1.0

    def faulted(self, job_id, seed=-1):
        self.faulted_jobs.append(job_id)


def job(payload_overrides=None, model=None):
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
    return {
        "id": f"test-{payload['sampler_name']}-{payload['seed']}",
        "model": model or DEFAULT_MODEL,
        "payload": payload,
    }


def asset(name):
    return Image.open(os.path.join(HERE, "assets", name)).convert("RGB")


def main():
    global DEFAULT_MODEL
    W.setup_logging(False)
    cfg = W.load_config(os.path.join(HERE, "bridgeData.yaml"))
    DEFAULT_MODEL = cfg["models"][0]
    w = W.Worker(cfg, HERE)
    w.client = StubClient()

    fails = []

    def check(name, ok, detail=""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            fails.append(name)

    print(f"using model {DEFAULT_MODEL!r} from bridgeData.yaml")

    # 1. happy path
    w.handle(job())
    r = w.client.submitted[-1]
    check("happy path", r["image"].size == (512, 512) and r["state"] is None,
          f"saved={r['image'].size} state={r['state']}")

    # 2. negative prompt via ###
    w.handle(job({"prompt": "a cat###blurry, watermark", "seed": 43}))
    check("### negative-prompt split handled", True)

    # 3. a non-512 size (the horde can send anything <= max_pixels)
    w.handle(job({"width": 512, "height": 384, "seed": 44, "sampler_name": "ddim"}))
    r = w.client.submitted[-1]
    check("non-square size", r["image"].size == (512, 384), f"{r['image'].size}")

    # 4. karras + another sampler
    w.handle(job({"sampler_name": "k_dpmpp_2m", "karras": True, "seed": 45}))
    check("karras + k_dpmpp_2m handled", True)

    # 5. unsupported sampler -> faulted, not submitted
    before = len(w.client.submitted)
    w.handle(job({"sampler_name": "not_a_sampler", "seed": 46}))
    check("unsupported sampler faulted",
          len(w.client.submitted) == before and bool(w.client.faulted_jobs),
          f"faulted={w.client.faulted_jobs}")

    # 6. a model we do not serve -> faulted
    before = len(w.client.submitted)
    w.handle(job({"seed": 47}, model="Some Model We Do Not Serve"))
    check("unknown model faulted",
          len(w.client.submitted) == before, f"faulted={w.client.faulted_jobs[-1:]}")

    # ---- censor wiring -------------------------------------------------------
    # Stub the classifier so the tests do not depend on generating an NSFW image;
    # what is under test here is *which reason* fires and *which placeholder* is used.
    real_is_nsfw = w.safety.is_nsfw
    w.safety.is_nsfw = lambda _img: True  # noqa: ARG005
    try:
        # 6a. job asks for censoring -> the "requested" placeholder
        w.handle(job({"use_nsfw_censor": True, "seed": 48}))
        r = w.client.submitted[-1]
        check("censor: job requests -> sfw_request placeholder",
              r["state"] == "censored" and r["image"].tobytes() == asset("nsfw_censor_sfw_request.png").tobytes(),
              f"state={r['state']}")

        # 6b. prompt matches censorlist -> the censorlist placeholder
        w.censorlist = ["astronaut"]
        w.handle(job({"use_nsfw_censor": True, "seed": 49}))
        r = w.client.submitted[-1]
        check("censor: censorlist wins over the job flag",
              r["state"] == "censored" and r["image"].tobytes() == asset("nsfw_censor_censorlist.png").tobytes(),
              f"state={r['state']}")
        w.censorlist = []

        # 6c. censor_nsfw on an sfw:false worker -> the sfw_worker placeholder
        w.censor_nsfw = True
        w.cfg["nsfw"] = False
        w.handle(job({"seed": 50}))  # job itself does not ask
        r = w.client.submitted[-1]
        check("censor: censor_nsfw on nsfw:false worker",
              r["state"] == "censored" and r["image"].tobytes() == asset("nsfw_censor_sfw_worker.png").tobytes(),
              f"state={r['state']}")
        w.censor_nsfw = False
        w.cfg["nsfw"] = True

        # 6d. no reason -> not censored, even though the classifier says nsfw
        w.handle(job({"seed": 51}))
        r = w.client.submitted[-1]
        check("censor: no reason -> untouched", r["state"] is None, f"state={r['state']}")
    finally:
        w.safety.is_nsfw = real_is_nsfw

    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
