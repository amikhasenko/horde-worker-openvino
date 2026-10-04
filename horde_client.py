"""Minimal AI Horde generation-worker client (pop / submit).

Field names are taken from the working worker's ``worker/jobs/poppers.py`` and
``worker/jobs/stable_diffusion.py`` so the payloads match what the server expects.
"""
from __future__ import annotations

import base64
import io
import logging

import requests
from PIL import Image

log = logging.getLogger("ov-sd.horde")

BRIDGE_VERSION = 24
# "Software Contact" reported to the Horde: agent~engine:version:repo. Points at this
# worker's own repository, not the reference worker's.
BRIDGE_AGENT = f"AI Horde Worker~openvino:{BRIDGE_VERSION}:https://github.com/amikhasenko/horde-worker-openvino"


class HordeClient:
    def __init__(
        self,
        api_key: str,
        worker_name: str,
        models: list[str],
        horde_url: str = "https://aihorde.net",
        max_power: int = 8,
        threads: int = 1,
        nsfw: bool = True,
        allow_img2img: bool = False,
        allow_painting: bool = False,
        allow_post_processing: bool = False,
        allow_controlnet: bool = False,
        allow_lora: bool = False,
        priority_usernames: list[str] | None = None,
        blacklist: list[str] | None = None,
        require_upfront_kudos: bool = False,
        allow_unsafe_ipaddr: bool = True,
        allow_sdxl_controlnet: bool = False,
        extra_slow_worker: bool = True,
        limit_max_steps: bool = True,
        amount: int = 1,
        timeout: int = 60,
    ):
        self.horde_url = horde_url.rstrip("/")
        self.worker_name = worker_name
        self.models = list(models)
        self.headers = {"apikey": api_key} if api_key else {}
        self.timeout = timeout
        # max_pixels caps the resolution the server will send us: 64*64*8*max_power.
        # max_power=8 -> 262144 px = 512x512.
        self.max_pixels = 64 * 64 * 8 * max_power
        # Field names follow horde_sdk.ai_horde_api.apimodels.ImageGenerateJobPopRequest
        # (the source of truth) — note `allow_unsafe_ipaddr`, not the legacy
        # `allow_unsafe_ip` the old worker still sends.
        self.pop_payload = {
            "name": worker_name,
            "max_pixels": self.max_pixels,
            "priority_usernames": priority_usernames or [],
            "nsfw": nsfw,
            "blacklist": blacklist or [],
            "models": self.models,
            "allow_img2img": allow_img2img,
            "allow_painting": allow_painting,
            "allow_unsafe_ipaddr": allow_unsafe_ipaddr,
            "threads": threads,
            "allow_post_processing": allow_post_processing,
            "allow_controlnet": allow_controlnet,
            "allow_sdxl_controlnet": allow_sdxl_controlnet,
            "allow_lora": allow_lora,
            "require_upfront_kudos": require_upfront_kudos,
            # Tell the server we are a slow worker so it only sends us jobs from users
            # who opted in to extra-slow processing. Pair with limit_max_steps so we
            # don't get wedged on an enormous job.
            "extra_slow_worker": extra_slow_worker,
            "limit_max_steps": limit_max_steps,
            "amount": amount,
            "bridge_version": BRIDGE_VERSION,
            "bridge_agent": BRIDGE_AGENT,
        }

    def pop(self) -> dict | None:
        r = requests.post(
            f"{self.horde_url}/api/v2/generate/pop",
            json=self.pop_payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        if r.status_code == 403:
            raise PermissionError(f"pop forbidden (check api_key / worker name): {r.text[:200]}")
        r.raise_for_status()
        job = r.json()
        if not job.get("id"):
            if job.get("message"):
                log.debug("no job: %s", job["message"])
            return None
        return job

    def submit(self, job_id: str, image: Image.Image, seed: int, state: str | None = None,
               quality: int = 95) -> float:
        buf = io.BytesIO()
        image.save(buf, format="WebP", quality=quality, method=6)
        body = {
            "id": job_id,
            "generation": base64.b64encode(buf.getvalue()).decode("utf8"),
            # `seed` must be an int and `state` is a *required* enum
            # ('' | ok | censored | faulted | csam | ...). Omitting `state` on success
            # is a 400.
            "seed": int(seed),
            "state": state or "ok",
        }
        if state == "censored":
            # ImageGenerationJobSubmitRequest carries both `state` (str) and `censored` (bool).
            body["censored"] = True
        return self._post_submit(body)

    def _post_submit(self, body: dict) -> float:
        r = requests.post(
            f"{self.horde_url}/api/v2/generate/submit",
            json=body,
            headers=self.headers,
            timeout=self.timeout,
        )
        if not r.ok:
            # The 4xx body carries the real reason; a bare raise_for_status does not.
            raise RuntimeError(f"submit -> HTTP {r.status_code}: {r.text[:400]}")
        return float(r.json().get("reward", 0))

    def faulted(self, job_id: str, seed: int = -1) -> None:
        body = {"id": job_id, "generation": "faulted", "seed": int(seed), "state": "faulted"}
        try:
            self._post_submit(body)
        except Exception as e:  # noqa: BLE001
            log.warning("could not report fault for %s: %s", job_id, e)
