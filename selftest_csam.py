"""Test the ported CSAM logic: one real end-to-end run plus unit tests of the decision
rule (which needs controlled similarities to exercise the >=3 / critical-tripling paths).

    .venv-sd/bin/python -s selftest_csam.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import csam  # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


class FakeClip:
    """Returns preset similarities for the CSAM word list."""

    def __init__(self, values):
        self.values = values

    def similarity(self, image, texts):
        return {t: self.values.get(t, 0.10) for t in texts}


def main():
    # ---- 1. real end-to-end on a benign image/prompt -------------------------
    from PIL import Image

    from ov_clip import OVClip

    img_path = os.path.join(os.path.dirname(HERE), "tmp", "waifu.png")
    if os.path.exists(img_path):
        clip = OVClip(os.path.join(os.path.dirname(HERE), "cache", "ov"),
                      os.path.join(os.path.dirname(HERE), "cache"), device="CPU")
        img = Image.open(img_path)
        model_info = {"nsfw": False, "tags": ["anime"]}
        is_csam, sims, details = csam.check_for_csam(clip, img, "1girl, portrait, blue hair, anime", model_info)
        check("real benign image -> not csam", not is_csam,
              f"(uc={len(details['found_uc'])} lewd={len(details['found_lewd'])})")
    else:
        print(f"[SKIP] real image test ({img_path} missing)")

    # ---- 2. unit tests of the decision rule ----------------------------------
    benign = {}

    # 2a. three underage signals + one lewd -> fires
    vals = {"child": 0.20, "toddler": 0.20, "infant": 0.20, "porn": 0.25}
    is_csam, _, d = csam.check_for_csam(FakeClip(vals), None, "a picture", benign)
    check("3 underage + lewd -> csam", is_csam, f"(uc={len(d['found_uc'])})")

    # 2b. two underage signals + lewd -> does NOT fire
    vals = {"child": 0.20, "toddler": 0.20, "porn": 0.25}
    is_csam, _, d = csam.check_for_csam(FakeClip(vals), None, "a picture", benign)
    check("2 underage + lewd -> not csam", not is_csam, f"(uc={len(d['found_uc'])})")

    # 2c. underage signals but no lewd -> does NOT fire
    vals = {"child": 0.20, "toddler": 0.20, "infant": 0.20}
    is_csam, _, _ = csam.check_for_csam(FakeClip(vals), None, "a picture", benign)
    check("underage but no lewd -> not csam", not is_csam)

    # 2d. one CRITICAL underage hit counts triple -> fires with a single lewd
    vals = {"child": 0.24, "porn": 0.25}  # 0.24 > UNDERAGE_CRITICAL['child']=0.225
    is_csam, _, d = csam.check_for_csam(FakeClip(vals), None, "a picture", benign)
    check("critical hit triples -> csam", is_csam, f"(uc entries={len(d['found_uc'])})")

    # 2e. model adjustments: an nsfw model pushes a borderline lewd word over
    vals = {"child": 0.20, "toddler": 0.20, "infant": 0.20, "nudity": 0.18}  # < 0.195
    is_csam, _, _ = csam.check_for_csam(FakeClip(vals), None, "a picture", {"nsfw": False, "tags": []})
    before = is_csam
    is_csam, _, _ = csam.check_for_csam(FakeClip(vals), None, "a picture", {"nsfw": True, "tags": []})
    check("model nsfw flag lifts lewd over threshold", (not before) and is_csam)

    print(f"\n{'ALL PASS' if not FAILS else 'FAILURES: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
