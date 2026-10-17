"""A stand-in for ``QwenImage21Pipeline`` that draws placeholder images.

It accepts the same call arguments the engine uses, so every code path above the model
(prompt building, selection handling, jobs, API, CLI, web UI) can run without weights.
Enable it with ``QWEN_IMAGE_ENABLE_MOCK=1`` and the ``mock`` variant.
"""

from __future__ import annotations

import math
import os
import random
import textwrap
import time
from types import SimpleNamespace

from PIL import Image, ImageDraw, ImageOps

from .selection import ANNOTATION_COLOR


def _dims(width, height, image, resolution):
    if image:
        w, h = image[-1].size
        ratio = w / h
        width = width or round(resolution * math.sqrt(ratio) / 32) * 32
        height = height or round(resolution / math.sqrt(ratio) / 32) * 32
    return (width or resolution) // 32 * 32, (height or resolution) // 32 * 32


def _red_region(img: Image.Image) -> Image.Image | None:
    """Filled interior of a red annotation outline, if the image carries one."""
    rgb = img.convert("RGB")
    r, g, b = rgb.split()
    red = Image.eval(r, lambda v: 255 if v >= ANNOTATION_COLOR[0] - 5 else 0)
    not_green = Image.eval(g, lambda v: 255 if v <= 10 else 0)
    not_blue = Image.eval(b, lambda v: 255 if v <= 10 else 0)
    from PIL import ImageChops

    band = ImageChops.multiply(ImageChops.multiply(red, not_green), not_blue)
    bbox = band.getbbox()
    if not bbox:
        return None
    # Flood the outside from the corners; whatever is left unflooded is inside the outline.
    fill = band.copy()
    w, h = fill.size
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if fill.getpixel(seed) == 0:
            ImageDraw.floodfill(fill, seed, 128)
    return fill.point(lambda v: 0 if v == 128 else 255)


class MockPipeline:
    def __init__(self):
        self.delay = float(os.environ.get("QWEN_IMAGE_MOCK_DELAY", "0.02"))
        self._interrupt = False

    @property
    def interrupt(self):
        return self._interrupt

    def __call__(
        self,
        prompt=None,
        image=None,
        width=None,
        height=None,
        num_inference_steps=40,
        generator=None,
        num_images_per_prompt=1,
        output_resolution=1024,
        callback_on_step_end=None,
        **_,
    ):
        self._interrupt = False
        width, height = _dims(width, height, image, output_resolution)
        seed = generator.initial_seed() if generator is not None else 0
        steps = num_inference_steps or 40
        for i in range(steps):
            if self._interrupt:
                break
            time.sleep(self.delay)
            if callback_on_step_end:
                callback_on_step_end(self, i, 1000 - i * 1000 // steps, {})

        transparent = "transparen" in (prompt or "").lower()
        outs = []
        for n in range(num_images_per_prompt):
            rng = random.Random(seed + n)
            if image:
                src = image[-1].convert("RGBA").resize((width, height))
                region = _red_region(image[-1])
                base = ImageOps.posterize(src.convert("RGB"), 4).convert("RGBA")
                if transparent:
                    alpha = (region.resize((width, height)) if region else None) or Image.new("L", (width, height), 0)
                    if not region:
                        ImageDraw.Draw(alpha).ellipse([width * 0.2, height * 0.2, width * 0.8, height * 0.8], fill=255)
                    base.putalpha(alpha)
            else:
                c1 = tuple(rng.randrange(256) for _ in range(3))
                c2 = tuple(rng.randrange(256) for _ in range(3))
                grad = Image.linear_gradient("L").resize((width, height))
                base = Image.composite(Image.new("RGB", (width, height), c1), Image.new("RGB", (width, height), c2), grad)
                base = base.convert("RGBA")
                if transparent:
                    alpha = Image.new("L", (width, height), 0)
                    ImageDraw.Draw(alpha).ellipse([width * 0.15, height * 0.15, width * 0.85, height * 0.85], fill=255)
                    base.putalpha(alpha)
            draw = ImageDraw.Draw(base)
            label = textwrap.fill(f"MOCK seed={seed + n}\n{prompt or ''}"[:400], 60)
            draw.multiline_text((12, 12), label, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 255))
            outs.append(base)
        return SimpleNamespace(images=outs)

    def set_progress_bar_config(self, **_):
        pass
