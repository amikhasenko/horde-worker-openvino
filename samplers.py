"""Horde sampler names -> diffusers schedulers.

The Horde only ever sends the 15 names in hordelib's ``SAMPLERS_MAP`` (the server
validates against that same list), so this mirrors it.  Where diffusers has no exact
equivalent we approximate, exactly as stock does for ``plms -> euler``.

Anything outside the list is unsupported: we fault the job rather than invent a sampler.
"""
from __future__ import annotations

import inspect

import diffusers

# horde name -> diffusers scheduler class name
SAMPLER_MAP = {
    "k_euler": "EulerDiscreteScheduler",
    "k_euler_a": "EulerAncestralDiscreteScheduler",
    "k_heun": "HeunDiscreteScheduler",
    "k_dpm_2": "KDPM2DiscreteScheduler",
    "k_dpm_2_a": "KDPM2AncestralDiscreteScheduler",
    "k_lms": "LMSDiscreteScheduler",
    "k_dpmpp_2s_a": "DPMSolverSinglestepScheduler",
    "k_dpmpp_sde": "DPMSolverSDEScheduler",
    "k_dpmpp_2m": "DPMSolverMultistepScheduler",
    "ddim": "DDIMScheduler",
    "uni_pc": "UniPCMultistepScheduler",
    "uni_pc_bh2": "UniPCMultistepScheduler",  # bh2 variant has no diffusers equivalent
    "plms": "PNDMScheduler",  # stock maps plms -> euler; PNDM is the closer name
    "k_dpm_fast": "DPMSolverMultistepScheduler",  # approximate
    "k_dpm_adaptive": "DPMSolverMultistepScheduler",  # approximate
}

# Which of those are exact vs approximated (for logging / honest accounting).
APPROXIMATED = {"uni_pc_bh2", "k_dpm_fast", "k_dpm_adaptive", "plms"}

DEFAULT_SAMPLER = "k_euler"
SUPPORTED = set(SAMPLER_MAP)


class UnsupportedSampler(ValueError):
    pass


def build_scheduler(name: str, base_config, karras: bool = False):
    """Return a scheduler instance for a Horde sampler name, built from the pipeline's
    existing scheduler config (so beta schedule / prediction type are inherited)."""
    if name not in SAMPLER_MAP:
        raise UnsupportedSampler(name)
    cls = getattr(diffusers, SAMPLER_MAP[name])
    cfg = dict(base_config)
    accepts_karras = "use_karras_sigmas" in inspect.signature(cls.__init__).parameters
    if karras and accepts_karras:
        try:
            return cls.from_config(cfg, use_karras_sigmas=True)
        except Exception:  # noqa: BLE001 - fall back to the plain variant
            pass
    return cls.from_config(cfg)
