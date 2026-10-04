"""OpenVINO SD1.5 generation ("dreamer") worker for the AI Horde.

Runs SD1.5 models on the Intel iGPU through OpenVINO — no torch/CUDA, no ComfyUI.
This is a slow-worker configuration: ~1.5 images/hour at 512x512, 20 steps on a UHD iGPU,
so it advertises a 512x512 pixel cap and one thread.

    ./run.sh
    python worker.py --config bridgeData.yaml

Protocol field names are mirrored from the reference worker (see horde_client.py). Only
SD1.5 text-to-image is supported; LoRA/ControlNet/inpaint/hires-fix are not advertised.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler

import yaml
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from horde_client import HordeClient  # noqa: E402
from pipeline import SDPipelineManager, UnsupportedModel  # noqa: E402
from samplers import DEFAULT_SAMPLER, SUPPORTED, UnsupportedSampler, build_scheduler  # noqa: E402
from safety import SafetyChecker  # noqa: E402

log = logging.getLogger("ov-sd")

_ASSETS = os.path.join(HERE, "assets")
# Placeholder images, one per censor reason — the same four the reference worker uses.
CENSOR_IMAGES = {
    "Requested": os.path.join(_ASSETS, "nsfw_censor_sfw_request.png"),
    "SFW worker": os.path.join(_ASSETS, "nsfw_censor_sfw_worker.png"),
    "Censorlist": os.path.join(_ASSETS, "nsfw_censor_censorlist.png"),
}
CSAM_CENSOR_IMAGE = os.path.join(_ASSETS, "nsfw_censor_csam.png")
MODEL_REFERENCE = "horde_model_reference/legacy/stable_diffusion.json"


def _load_image(path: str):
    return Image.open(path).convert("RGB") if os.path.exists(path) else None


def setup_logging(verbose: bool = False):
    log.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    log.addHandler(sh)
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    fh = RotatingFileHandler(os.path.join(HERE, "logs", "worker.log"), maxBytes=5 << 20, backupCount=3)
    fh.setFormatter(fmt)
    log.addHandler(fh)


def split_prompt(prompt: str) -> tuple[str, str]:
    """The Horde packs the negative prompt onto the positive with a '###' separator."""
    if "###" in prompt:
        pos, neg = prompt.split("###", 1)
        return pos, neg
    return prompt, ""


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class Worker:
    def __init__(self, cfg: dict, root: str):
        self.cfg = cfg
        self.root = root
        self.ov_dir = cfg.get("ov_dir", os.path.join(cfg["cache_home"], "ov"))
        models = cfg["models"]
        ov_models = {name: os.path.join(cfg["ov_sd_dir"], cfg["model_dirs"][name]) for name in models}
        self.client = HordeClient(
            api_key=cfg.get("api_key") or os.environ.get("HORDE_API_KEY", ""),
            worker_name=cfg["dreamer_name"],
            models=list(models),
            horde_url=cfg.get("horde_url", "https://aihorde.net"),
            max_power=int(cfg.get("max_power", 8)),
            threads=int(cfg.get("max_threads", 1)),
            nsfw=bool(cfg.get("nsfw", True)),
            allow_img2img=bool(cfg.get("allow_img2img", False)),
            allow_painting=bool(cfg.get("allow_painting", False)),
            allow_post_processing=bool(cfg.get("allow_post_processing", False)),
            allow_controlnet=bool(cfg.get("allow_controlnet", False)),
            allow_lora=bool(cfg.get("allow_lora", False)),
            priority_usernames=cfg.get("priority_usernames") or [],
            blacklist=cfg.get("blacklist") or [],
            require_upfront_kudos=bool(cfg.get("require_upfront_kudos", False)),
            allow_unsafe_ipaddr=bool(cfg.get("allow_unsafe_ipaddr", True)),
            allow_sdxl_controlnet=False,
            # Slow worker: the server only sends jobs from users who opted into
            # extra-slow processing; limit_max_steps stops us getting wedged.
            extra_slow_worker=bool(cfg.get("extra_slow_worker", True)),
            limit_max_steps=bool(cfg.get("limit_max_steps", True)),
            amount=1,
        )
        self.models = models
        self.pipes = SDPipelineManager(
            ov_models,
            device=cfg.get("device", "GPU"),
            max_cached=int(cfg.get("max_cached_pipelines", 2)),
        )
        self.safety = SafetyChecker(self.ov_dir, device="CPU")
        self.censor_images = {reason: _load_image(path) for reason, path in CENSOR_IMAGES.items()}
        self.csam_censor_image = _load_image(CSAM_CENSOR_IMAGE)
        # Censoring policy, mirroring the reference worker's bridge data.
        #   censor_nsfw : censor NSFW results even when the job did not ask (only has an
        #                 effect on a worker advertising nsfw: false)
        #   censorlist  : prompts containing any of these words force the censor check
        self.censor_nsfw = bool(cfg.get("censor_nsfw", False))
        self.censorlist = [w for w in (cfg.get("censorlist") or []) if w]
        self.enable_csam = bool(cfg.get("enable_csam", True))
        self.csam_device = cfg.get("csam_device", "CPU")
        self.model_info = self._load_model_info()
        self._clip = None
        self.jobs_done = 0
        self.kudos_total = 0.0

    def _load_model_info(self) -> dict:
        """Per-model ``nsfw`` flag and ``tags`` from the Horde model reference — the CSAM
        heuristic adjusts its thresholds with both."""
        import json

        path = os.path.join(self.cfg["cache_home"], MODEL_REFERENCE)
        if not os.path.exists(path):
            log.warning("model reference not found at %s; CSAM model adjustments disabled", path)
            return {}
        ref = json.loads(open(path, encoding="utf-8").read())
        return {
            name: {"nsfw": ref.get(name, {}).get("nsfw", False), "tags": ref.get(name, {}).get("tags") or []}
            for name in self.models
        }

    def _censor_placeholder(self, reason: str) -> Image.Image:
        """The placeholder for a censor reason, or black if the asset is missing."""
        return self.censor_images.get(reason) or Image.new("RGB", (512, 512), (0, 0, 0))

    def _get_clip(self):
        """Lazily build the CLIP similarity model (CPU by default: keep the iGPU for SD)."""
        if self._clip is None:
            from ov_clip import OVClip

            log.info("loading CLIP for the CSAM check on %s", self.csam_device)
            self._clip = OVClip(self.ov_dir, self.cfg["cache_home"], device=self.csam_device)
        return self._clip

    # ------------------------------------------------------------------ one job
    def handle(self, job: dict) -> None:
        job_id = job["id"]
        model = job["model"]
        payload = job["payload"]
        seed = int(payload["seed"])
        width = int(payload["width"])
        height = int(payload["height"])
        steps = int(payload["ddim_steps"])
        cfg_scale = float(payload["cfg_scale"])
        sampler = payload.get("sampler_name", DEFAULT_SAMPLER)
        karras = bool(payload.get("karras", False))
        prompt, neg = split_prompt(payload["prompt"])

        log.info(
            "job %s | %s %dx%d %d steps cfg %s %s%s | %r",
            job_id, model, width, height, steps, cfg_scale, sampler,
            " karras" if karras else "", prompt[:80],
        )
        if payload.get("clip_skip") not in (None, 1):
            log.warning("job %s requests clip_skip=%s; not supported by the OV export, ignoring",
                        job_id, payload.get("clip_skip"))
        if payload.get("hires_fix"):
            log.warning("job %s requests hires_fix; not supported, generating base image only", job_id)

        if model not in self.models:
            log.error("job %s asked for model %r which we do not serve", job_id, model)
            self.client.faulted(job_id, seed)
            return
        if sampler not in SUPPORTED:
            log.error("job %s asked for unsupported sampler %r -> faulting (stock would refuse too)",
                      job_id, sampler)
            self.client.faulted(job_id, seed)
            return

        import torch

        try:
            pipe = self.pipes.get(model, width, height)
        except UnsupportedModel:
            self.client.faulted(job_id, seed)
            return

        base_cfg = getattr(pipe, "base_scheduler_config", None) or dict(pipe.scheduler.config)
        pipe.scheduler = build_scheduler(sampler, base_cfg, karras)
        gen = torch.Generator().manual_seed(seed)

        steps = max(1, min(steps, 150))
        cfg_scale = max(0.0, min(cfg_scale, 30.0))

        t = time.time()
        result = pipe(
            prompt,
            negative_prompt=neg or None,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            height=height,
            width=width,
            generator=gen,
            output_type="pil",
        )
        image = result.images[0]
        gen_time = time.time() - t

        # Censor decision, mirroring the reference worker (worker/jobs/stable_diffusion.py):
        # three reasons, each with its own placeholder image.  Order matters — the
        # censorlist check takes precedence over the job's own flag for image choice.
        # Note stock's censorlist does *not* bypass the NSFW classifier: it only forces
        # the check to run and selects a different placeholder.
        censor_reason = None
        if self.censor_nsfw and not self.cfg.get("nsfw", True):
            censor_reason = "SFW worker"
        if any(word in prompt for word in self.censorlist):
            censor_reason = "Censorlist"
        elif payload.get("use_nsfw_censor", False):
            censor_reason = "Requested"

        state = None
        if censor_reason and self.safety.is_nsfw(image):
            log.info("job %s censored (nsfw); reason=%s", job_id, censor_reason)
            # Deviation from stock (deliberate): stock pastes the 512x512 placeholder
            # regardless of the job's size; we resize it so a non-square job returns a
            # correctly-sized image.
            image = self._censor_placeholder(censor_reason).resize(image.size)
            state = "censored"

        # Stock skips the CSAM check when the image was already censored.
        if self.enable_csam and not state:
            import csam as csam_mod

            is_csam, _sims, details = csam_mod.check_for_csam(
                self._get_clip(), image, payload["prompt"], self.model_info.get(model, {}),
            )
            if is_csam:
                log.warning(
                    "job %s CENSORED AS CSAM | %d underage signals %s | lewd %s",
                    job_id,
                    len(details["found_uc"]),
                    sorted({e["word"] for e in details["found_uc"]}),
                    sorted({e["word"] for e in details["found_lewd"]}),
                )
                image = (self.csam_censor_image or Image.new("RGB", image.size, (0, 0, 0))).resize(image.size)
                state = "csam"

        reward = self.client.submit(job_id, image, seed, state)
        self.jobs_done += 1
        self.kudos_total += reward
        log.info(
            "job %s done in %.1fs | reward %.2f | total %.1f kudos over %d jobs",
            job_id, gen_time, reward, self.kudos_total, self.jobs_done,
        )

    # ------------------------------------------------------------------ loop
    def run(self):
        log.info("worker %r starting | models=%s | max_pixels=%d (%s)",
                 self.client.worker_name, self.models, self.client.max_pixels, self.cfg.get("device", "GPU"))
        while True:
            try:
                job = self.client.pop()
            except PermissionError as e:
                log.error("%s", e)
                return 1
            except Exception as e:  # noqa: BLE001
                log.warning("pop failed (%s); retrying", e)
                time.sleep(5)
                continue
            if not job:
                time.sleep(1)
                continue
            try:
                self.handle(job)
            except KeyboardInterrupt:
                raise
            except Exception:  # noqa: BLE001
                log.exception("job %s failed", job.get("id"))
                try:
                    self.client.faulted(job["id"], int(job.get("payload", {}).get("seed", -1)))
                except Exception:  # noqa: BLE001
                    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "bridgeData.yaml"))
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    cfg = load_config(args.config)
    worker = Worker(cfg, HERE)
    try:
        return worker.run()
    except KeyboardInterrupt:
        log.info("interrupted; %d jobs, %.1f kudos this session", worker.jobs_done, worker.kudos_total)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
