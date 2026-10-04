"""OpenVINO SD1.5 pipeline manager: load, reshape to the job's size, compile, cache.

The exported IR is dynamic in H/W, so any resolution works without re-exporting — but
``reshape()`` mutates the pipeline in place, so each (model, width, height) needs its own
pipeline object.  Compiled pipelines are therefore held in a small LRU: each one is
~2 GB of weights in (shared) GPU memory, so the cap matters on a 16 GB box.
"""
from __future__ import annotations

import logging
import time
from collections import OrderedDict

log = logging.getLogger("ov-sd.pipeline")


class UnsupportedModel(KeyError):
    pass


class SDPipelineManager:
    def __init__(self, models: dict[str, str], device: str = "GPU", max_cached: int = 2):
        """``models`` maps a Horde model name to its exported OpenVINO directory."""
        self.models = dict(models)
        self.device = device
        self.max_cached = max(1, max_cached)
        self._cache: OrderedDict[tuple[str, int, int], object] = OrderedDict()

    def available(self) -> list[str]:
        return list(self.models)

    def _build(self, model: str, width: int, height: int):
        from optimum.intel import OVStableDiffusionPipeline

        if model not in self.models:
            raise UnsupportedModel(model)
        t = time.time()
        pipe = OVStableDiffusionPipeline.from_pretrained(self.models[model])
        pipe.to(device=self.device)
        pipe.reshape(batch_size=1, height=height, width=width, num_images_per_prompt=1)
        pipe.compile()
        # Remember the *exported* scheduler config: jobs replace pipe.scheduler, so reading
        # pipe.scheduler.config later would drift to whatever the previous job set.
        pipe.base_scheduler_config = dict(pipe.scheduler.config)
        log.info(
            "compiled %s @ %dx%d on %s in %.1fs",
            model, width, height, self.device, time.time() - t,
        )
        return pipe

    def get(self, model: str, width: int, height: int):
        key = (model, width, height)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        pipe = self._build(model, width, height)
        self._cache[key] = pipe
        while len(self._cache) > self.max_cached:
            old_key, old = self._cache.popitem(last=False)
            log.info("evicting compiled pipeline %s", old_key)
            del old
        return pipe

    def unload_all(self):
        self._cache.clear()
