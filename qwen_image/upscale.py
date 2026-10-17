"""AI upscaling with dedicated super-resolution models (Real-ESRGAN and friends, via spandrel).

Qwen-Image-2.1 generates at most about 2048 px, so on its own it cannot enlarge an image by
2.5x or 10x. Super-resolution models can: they enlarge by a fixed factor (4x or 2x) and invent
plausible fine detail. Images are processed in overlapping tiles, so the input size is limited
only by RAM. Any factor or target size is reached in passes:

1. The first pass always runs on the original, so no source detail is thrown away.
2. Later passes first shrink the intermediate so that the model lands exactly on the target,
   which avoids huge intermediates (10x is 4x, then 2.5x shrunk-to-fit 4x).
3. A final Lanczos resize hits the exact pixel size.

Optionally the engine first runs a Qwen "restore and enhance" pass (see ``Engine.upscale``).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

MAX_SIDE = 16384
MAX_SCALE = 16.0


@dataclass(frozen=True)
class UpscaleModel:
    name: str
    label: str
    description: str
    repo: str
    filename: str
    scale: int
    license: str
    approx_download_mb: int

    def to_dict(self) -> dict:
        return asdict(self)


UPSCALERS = {
    m.name: m
    for m in [
        UpscaleModel(
            "realesrgan-x4plus", "Real-ESRGAN x4+ (general)",
            "Good all-rounder for photos, renders and illustrations; removes blur, noise and JPEG artefacts.",
            "Comfy-Org/Real-ESRGAN_repackaged", "RealESRGAN_x4plus.safetensors", 4, "BSD-3-Clause", 64,
        ),
        UpscaleModel(
            "nomos-webphoto", "Nomos WebPhoto 4x (photos)",
            "RealPLKSR trained on real-world web photos: natural textures, fewer painterly artefacts.",
            "Phips/4xNomosWebPhoto_RealPLKSR", "4xNomosWebPhoto_RealPLKSR.safetensors", 4, "CC-BY-4.0", 30,
        ),
        UpscaleModel(
            "realesrgan-x4plus-anime", "Real-ESRGAN x4+ Anime (anime, cartoons)",
            "Tuned for anime, cartoons and flat-coloured art: crisp lines and clean fills.",
            "hlky/RealESRGAN_x4plus_anime_6B", "diffusion_pytorch_model.safetensors", 4, "BSD-3-Clause", 18,
        ),
        UpscaleModel(
            "realesrgan-x2", "Real-ESRGAN x2 (gentle)",
            "2x model that changes the image less; good for small enlargements of clean images.",
            "ai-forever/Real-ESRGAN", "RealESRGAN_x2.pth", 2, "BSD-3-Clause", 64,
        ),
    ]
}
DEFAULT_UPSCALER = "realesrgan-x4plus"


def get_upscaler(name: str | None) -> UpscaleModel:
    name = name or DEFAULT_UPSCALER
    if name not in UPSCALERS:
        raise ValueError(f"Unknown upscale model '{name}'; choose from {', '.join(UPSCALERS)}")
    return UPSCALERS[name]


def target_size(size: tuple[int, int], scale: float | None = None, width: int | None = None, height: int | None = None) -> tuple[int, int]:
    """The output size for a factor, or for a width and/or height (aspect kept; both = fit inside)."""
    w, h = size
    if width or height:
        k = min(width / w if width else math.inf, height / h if height else math.inf)
    elif scale:
        k = float(scale)
    else:
        raise ValueError("Give a scale factor, or a target width and/or height")
    if k <= 0:
        raise ValueError("The scale must be positive")
    if k > MAX_SCALE:
        raise ValueError(f"At most {MAX_SCALE:g}x is supported")
    out = max(1, round(w * k)), max(1, round(h * k))
    if max(out) > MAX_SIDE:
        raise ValueError(f"The result would be {out[0]}x{out[1]}; at most {MAX_SIDE} px per side is supported")
    return out


def plan_passes(size: tuple[int, int], target: tuple[int, int], model_scale: int) -> list[tuple[int, int]]:
    """The input size of each model pass (see the module docstring)."""
    passes: list[tuple[int, int]] = []
    if target[0] <= size[0] and target[1] <= size[1]:
        return passes
    cur = size
    while True:
        if passes:
            r = target[0] / cur[0]
            if r <= 1.15:  # close enough for a plain resize
                break
            if r < model_scale:
                cur = max(1, round(target[0] / model_scale)), max(1, round(target[1] / model_scale))
        passes.append(cur)
        cur = cur[0] * model_scale, cur[1] * model_scale
        if cur[0] >= target[0] and cur[1] >= target[1]:
            break
    return passes


def is_downloaded(model: UpscaleModel) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model.repo, model.filename), str)


def download(model: UpscaleModel, report: Callable[[str], None] | None = None) -> str:
    from huggingface_hub import hf_hub_download

    (report or log.info)(f"Downloading {model.repo}/{model.filename}")
    return hf_hub_download(model.repo, model.filename)


def _tiles(w: int, h: int, tile: int) -> int:
    return math.ceil(w / tile) * math.ceil(h / tile)


class _MockModel:
    """Bicubic stand-in so every code path above the weights runs in tests."""

    scale = 4

    def __call__(self, x):
        import torch.nn.functional as F

        return F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)


class Upscalers:
    """Lazily loaded super-resolution models; run on the GPU only for the duration of a call."""

    def __init__(self, mock: bool = False):
        self.mock = mock
        self._models: dict = {}
        self._lock = threading.Lock()

    def _get(self, spec: UpscaleModel, report: Callable[[str], None]):
        if self.mock:
            return _MockModel(), spec.scale, False, False
        with self._lock:
            if spec.name not in self._models:
                import spandrel

                path = download(spec, report if not is_downloaded(spec) else (lambda msg: None))
                report(f"Loading {spec.label}")
                d = spandrel.ModelLoader().load_from_file(path)
                if d.scale != spec.scale or d.input_channels != 3:
                    raise RuntimeError(f"{spec.name}: expected a 3-channel {spec.scale}x model, got {d.scale}x/{d.input_channels}")
                d.model.eval()
                self._models[spec.name] = (d.model, d.scale, d.supports_half, d.supports_bfloat16)
            return self._models[spec.name]

    def upscale(
        self,
        image: Image.Image,
        target: tuple[int, int],
        model: str | None = None,
        tile: int = 512,
        progress: Callable[[int, int], None] = lambda step, total: None,
        status: Callable[[str], None] = lambda msg: None,
        cancelled: Callable[[], bool] = lambda: False,
    ) -> tuple[Image.Image, dict]:
        spec = get_upscaler(model)
        tile = max(64, min(2048, int(tile)))
        alpha = image.getchannel("A") if "A" in image.getbands() else None
        cur = image.convert("RGB")
        passes = plan_passes(cur.size, target, spec.scale)
        info = {"model": spec.name, "passes": len(passes), "size": list(target), "source_size": list(image.size)}
        if passes:
            net, scale, half, bf16 = self._get(spec, status)
            total = sum(_tiles(w, h, tile) for w, h in passes)
            done = [0]

            def tick():
                done[0] += 1
                progress(done[0], total)
                if cancelled():
                    from .engine import Cancelled

                    raise Cancelled()

            for i, size in enumerate(passes):
                status(f"Upscaling with {spec.label}" + (f" (pass {i + 1}/{len(passes)})" if len(passes) > 1 else ""))
                if cur.size != size:
                    cur = cur.resize(size, Image.Resampling.LANCZOS)
                cur = self._run(net, scale, half, bf16, cur, tile, tick)
        if cur.size != target:
            cur = cur.resize(target, Image.Resampling.LANCZOS)
        if alpha is not None:
            cur.putalpha(alpha.resize(target, Image.Resampling.LANCZOS))
        return cur, info

    def _run(self, net, scale, half, bf16, image: Image.Image, tile: int, tick: Callable[[], None]) -> Image.Image:
        """One model pass over overlapping tiles; each tile's border is discarded to hide seams."""
        import torch

        device = "cuda" if torch.cuda.is_available() and not self.mock else "cpu"
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.float16 if half else torch.bfloat16 if bf16 else torch.float32
        src = np.array(image)
        h, w = src.shape[:2]
        out = np.empty((h * scale, w * scale, 3), np.uint8)
        pad = 16 if tile <= 256 else 32
        if hasattr(net, "to"):
            net.to(device=device, dtype=dtype)
        try:
            with torch.inference_mode():
                for y0 in range(0, h, tile):
                    for x0 in range(0, w, tile):
                        y1, x1 = min(h, y0 + tile), min(w, x0 + tile)
                        py0, px0, py1, px1 = max(0, y0 - pad), max(0, x0 - pad), min(h, y1 + pad), min(w, x1 + pad)
                        x = torch.from_numpy(np.ascontiguousarray(src[py0:py1, px0:px1])).to(device)
                        x = x.permute(2, 0, 1).unsqueeze(0).to(dtype) / 255.0
                        y = net(x).float().clamp_(0, 1).mul_(255).round_().byte()[0].permute(1, 2, 0).cpu().numpy()
                        oy, ox = (y0 - py0) * scale, (x0 - px0) * scale
                        out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = y[oy:oy + (y1 - y0) * scale, ox:ox + (x1 - x0) * scale]
                        tick()
        finally:
            if hasattr(net, "to") and device == "cuda":
                net.to("cpu")
                torch.cuda.empty_cache()
        return Image.fromarray(out)
