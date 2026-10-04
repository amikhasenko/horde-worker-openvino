# horde-worker-openvino

A [AI Horde](https://aihorde.net) image-generation ("dreamer") worker that runs **Stable
Diffusion 1.5 on an Intel iGPU through OpenVINO** — no CUDA, no ROCm, no NVIDIA, no
ComfyUI, no hordelib.

Existing Horde backends do not speak OpenVINO: the reference
[AI-Horde-Worker](https://github.com/Haidra-Org/AI-Horde-Worker) drives ComfyUI/hordelib
(PyTorch, CUDA-centric) and the newer
[horde-worker-reGen](https://github.com/Haidra-Org/horde-worker-reGen) targets CUDA,
ROCm and DirectML only. Neither can use an Intel integrated GPU. This worker talks the
Horde's job protocol directly and does inference with OpenVINO, so an ordinary laptop
iGPU can serve real image jobs.

It is a **slow worker**: on a UHD-class iGPU it produces roughly one 512×512 / 20-step
image per 40 seconds (~0.1 megapixelsteps/second). It therefore advertises itself as an
extra-slow worker, caps requests at 512×512, and earns kudos at the slow-worker rate.

> ## ⚠️ AI-generated code, published for license compliance
>
> Much of this codebase was written by an AI coding agent (Anthropic's Claude Code)
> under human direction and review. It is published openly to satisfy the AGPL it is
> received under: it derives from AI-Horde-Worker (AGPL-3.0) — `csam.py` is a verbatim
> port — so copies can only be conveyed under AGPL-3.0 with their source, and this
> repository is that source.
>
> See [`NOTICE`](NOTICE) for provenance and attribution. Read the code before running
> it, particularly the safety checks.

## How it works

```
            ┌──────────────┐   POST /api/v2/generate/pop
            │  AI Horde    │◄──────────────────────────────┐
            │   server     │                               │
            └──────┬───────┘   POST /api/v2/generate/submit│
                   │                                       │
        job (model, prompt, seed, w, h, steps, sampler)    │
                   ▼                                       │
   ┌───────────────────────────────────────────────────────┴────────┐
   │ worker.py                                                      │
   │                                                                │
   │  pipeline.py   OVStableDiffusionPipeline (optimum-intel)       │
   │                reshaped to the job's size, LRU-cached          │
   │      │                                                         │
   │      ├─ samplers.py   Horde sampler name -> diffusers scheduler│
   │      │                                                         │
   │      ▼                                                         │
   │   iGPU (OpenVINO, fp16 weights)                                │
   │      │                                                         │
   │      ▼                                                         │
   │  safety.py     OpenVINO safety checker (CPU) — NSFW censor     │
   │  csam.py       CLIP-similarity CSAM heuristic (CPU)            │
   │  ov_clip.py    CLIP ViT-L/14 image/text embeddings (CPU)       │
   └────────────────────────────────────────────────────────────────┘
```

Three models are exported to OpenVINO IR, all fp16 weights:

| IR | Purpose | Built by |
|----|---------|----------|
| `ov_sd/<model>/` | the SD1.5 pipeline itself | `tools/convert_sd_single.py` |
| `ov/clip_visual.xml`, `ov/clip_text.xml` | CLIP ViT-L/14 for the CSAM heuristic | `tools/convert_ov_assets.py` |
| `ov/safety_visual.xml` + `ov/safety_meta.npz` | NSFW safety checker | `tools/convert_ov_assets.py` |

Both safety models run on the **CPU** so the iGPU stays free for generation.

**fp16 for generation, fp32 for the safety/CLIP IR.** SD is exported with
`--weight-format fp16`, which is what every Horde generation worker runs (hordelib's
`half_precision=True` default) — not a degradation. The CLIP/safety IR is fp32 because
the CSAM heuristic compares embedding similarities against an fp32 reference.

## Requirements

- Linux, an Intel GPU with a working OpenCL or Level Zero runtime (`clinfo` should list
  an Intel platform; you may need `intel-opencl-icd` / `libze-intel-gpu1` and to be in
  the `render` and `video` groups).
- Python 3.10+.
- ~12 GB of RAM free and ~10 GB of disk (IR + OpenVINO compile cache, per model).
- An AI Horde API key — register at <https://aihorde.net/register>.

## Setup

```bash
git clone https://github.com/amikhasenko/horde-worker-openvino
cd horde-worker-openvino

python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

# 1. build the CLIP + safety-checker IR  (needs the conversion extras, see requirements.txt)
pip install pytorch_lightning git+https://github.com/openai/CLIP.git
python tools/convert_ov_assets.py --out "$PWD/cache/ov"

# 2. export at least one SD1.5 model
cp bridgeData_template.yaml bridgeData.yaml
$EDITOR bridgeData.yaml          # set cache_home/ov_dir/ov_sd_dir and your api_key
python add_model.py --name stable_diffusion

# 3. run
./run.sh
```

Prefer not to put the key in a file at all: leave `api_key: ""` in `bridgeData.yaml`
and export `HORDE_API_KEY` instead.

The worker is verified against the `dreamer_name` in the config — make it unique
horde-wide or the server rejects the registration.

## Configuration

`bridgeData.yaml` (copy it from `bridgeData_template.yaml`; it is git-ignored because it
holds your API key). The important keys:

| Key | Meaning |
|-----|---------|
| `dreamer_name` | this worker's unique name, shown on the Horde |
| `api_key` | your key, or `""` to use `HORDE_API_KEY` |
| `cache_home`, `ov_dir`, `ov_sd_dir` | where models and IR live |
| `models`, `model_dirs` | Horde model names, and their directory under `ov_sd_dir` |
| `device` | `GPU` (Intel iGPU) or `CPU` (fallback, several times slower) |
| `max_power` | `8` → `max_pixels = 64*64*8*8` = 262144 = 512×512 cap |
| `max_cached_pipelines` | compiled pipelines to keep; each is ~2 GB of shared memory |
| `extra_slow_worker`, `limit_max_steps` | sent to the server; mark the worker as slow and bound job length |
| `enable_csam`, `csam_device` | the CSAM heuristic and where its CLIP runs |

Capabilities that are *not* implemented (`img2img`, `painting`, `controlnet`, `lora`,
hires-fix, clip-skip, inpainting) are advertised as unsupported rather than faked. Jobs
arriving with an unknown sampler or a model we do not serve are **faulted**, the same
choice the reference worker makes.

## Adding models

```bash
python add_model.py --name "Deliberate" --dry-run   # look it up, download nothing
python add_model.py --name "Deliberate"             # download, convert, export
```

It looks the name up in the Horde model reference (exact match required — that string is
what the server sends), downloads the single-file checkpoint, converts it to a diffusers
directory and then to OpenVINO fp16, deletes the intermediates and prints the
`bridgeData.yaml` lines to add. **SD1.5 only**: the worker hardcodes SD1.5 semantics
(CLIP text encoder, 77-token context, 8× VAE downscale), so the script refuses anything
whose Horde baseline is not `stable diffusion 1`.

Roughly 2 GB of IR plus ~2 GB of OpenVINO compile cache per model.

## Testing

```bash
python selftest_worker.py     # drives synthetic jobs through the worker with a stub client
python selftest_csam.py       # unit-tests the CSAM decision rule + one real run
python tools/test_job.py      # submit a real job pinned to this worker, wait, fetch the image
python tools/bench_sd_ov.py --dir cache/ov_sd/stable_diffusion   # iGPU benchmark
```

`tools/test_job.py` is the end-to-end check: it resolves your worker's UUID, submits a
job pinned to it (with `extra_slow_workers`, without which a slow worker is never
eligible), polls to completion and writes the returned image to `tmp/`.

## Notes and limitations

- **Deterministic.** The same seed and prompt produce byte-identical images across jobs
  and worker restarts (verified by md5).
- `plms`, `k_dpm_fast`, `k_dpm_adaptive` and `uni_pc_bh2` have no exact diffusers
  equivalent and are approximated — the same kind of substitution stock makes
  (`plms → euler`).
- The CSAM check is a *similarity heuristic* ported verbatim from the reference worker
  (`len(underage_hits) >= 3 and any(lewd_hit)`, with critical terms counting triple), not
  a classifier. It is copied unchanged so it behaves identically — see the caveats in
  that file.
- Censoring replaces the image with a placeholder and reports `state: censored` /
  `csam` to the server; it does not prevent generation.
- One thread, one image at a time. This hardware cannot do more, and claiming otherwise
  would just queue jobs behind it.

## License

[AGPL-3.0-or-later](LICENSE). Because it is derived from AI-Horde-Worker (AGPL-3.0), it
cannot be relicensed permissively. See [`NOTICE`](NOTICE) for what came from where.
