"""Live previews of an image while it is being sampled.

At each step the scheduler has moved the latents from noise level ``sigma_i`` to ``sigma_{i+1}``
along the model's predicted velocity, so two consecutive latents give back that velocity and
with it the model's current estimate of the *finished* image (flow matching: ``x0 = x - sigma*v``).
That estimate is what gets shown, which looks far more like the final image than the raw noisy
latents would.

Two ways to turn it into pixels:

``fast``
    A linear map from the 64 latent channels to RGBA per 16x16 tile. It is fitted once per
    loaded model by VAE-encoding a synthetic calibration image, then costs almost nothing per
    step. The preview is low-resolution and colour-approximate.
``full``
    The real VAE decode. Exact, but slower, and with CPU offload it swaps the denoiser out of
    VRAM each time, so it only runs every few steps.
"""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Callable

from PIL import Image, ImageDraw

log = logging.getLogger(__name__)

MODES = ("off", "fast", "full")
DEFAULT_EVERY = {"fast": 1, "full": 5}
FAST_MAX_SIDE = 512


def output_dims(width, height, images, resolution) -> tuple[int, int]:
    """The (width, height) QwenImage21Pipeline will sample at, computed the same way it does."""
    if images and (width is None or height is None):
        w, h = images[-1].size
        ratio = w / h
        cw = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
        ch = round(cw / ratio / 32) * 32
        width, height = width or cw, height or ch
    width, height = width or resolution, height or resolution
    return width // 32 * 32, height // 32 * 32


def _calibration_image(size: int = 768) -> Image.Image:
    """Busy, colourful content so the least-squares fit sees a wide range of colours and alpha."""
    rng = random.Random(1234)
    img = Image.new("RGBA", (size, size))
    grad = Image.linear_gradient("L").resize((size, size))
    img.paste(Image.merge("RGBA", (grad, grad.rotate(90), grad.rotate(180), Image.new("L", (size, size), 255))))
    draw = ImageDraw.Draw(img)
    for _ in range(140):
        x0, y0 = rng.randrange(size), rng.randrange(size)
        x1, y1 = x0 + rng.randrange(16, size // 3), y0 + rng.randrange(16, size // 3)
        color = tuple(rng.randrange(256) for _ in range(3)) + (rng.choice([0, 128, 255, 255, 255]),)
        (draw.ellipse if rng.random() < 0.5 else draw.rectangle)([x0, y0, x1, y1], fill=color)
    return img


def _fit_fast_map(pipe):
    """Least-squares (64 + bias) -> RGBA map from VAE latents to 16x16-tile mean colours."""
    import numpy as np
    import torch

    cached = getattr(pipe, "_qi_preview_map", None)
    if cached is not None:
        return cached
    img = _calibration_image()
    tensor = pipe.image_processor.preprocess(img, width=img.width, height=img.height).unsqueeze(2)
    vae = pipe.vae
    with torch.no_grad():
        latents = pipe._encode_vae_image(tensor.to(pipe._execution_device, vae.dtype), None)  # [1, C, 1, h, w]
    c, h, w = latents.shape[1], latents.shape[3], latents.shape[4]
    x = latents[0, :, 0].float().reshape(c, h * w).T.cpu()  # [n, C]
    target = img.resize((w, h), Image.Resampling.BOX)
    y = torch.from_numpy(np.asarray(target, dtype=np.float32).reshape(-1, 4) / 255.0)  # [n, 4]
    x = torch.cat([x, torch.ones(x.shape[0], 1)], dim=1)
    weights = torch.linalg.lstsq(x, y).solution  # [C + 1, 4]
    pipe._qi_preview_map = weights
    return weights


class Previewer:
    """Called from the pipeline's step callback; hands finished preview images to ``emit``."""

    def __init__(self, pipe, mode: str, every: int | None, width: int, height: int, total: int,
                 emit: Callable[[Image.Image, int], None], mock: bool = False):
        if mode not in MODES:
            raise ValueError(f"preview must be one of {', '.join(MODES)}")
        self.pipe, self.mode, self.emit, self.mock = pipe, mode, emit, mock
        self.every = max(1, every or DEFAULT_EVERY.get(mode, 1))
        self.width, self.height, self.total = width, height, total
        self.grid = (height // 16, width // 16)
        self.prev = None
        self.failed = False
        if mode == "fast" and not mock:
            try:
                _fit_fast_map(pipe)
            except Exception:
                log.exception("Could not calibrate fast previews; disabling them")
                self.failed = True

    @property
    def enabled(self) -> bool:
        return self.mode != "off" and not self.failed

    def __call__(self, step: int, latents) -> None:
        """``step`` is 0-based; ``latents`` are the packed latents *after* that step."""
        if not self.enabled:
            return
        if self.mock:
            if (step + 1) % self.every == 0 or step + 1 == self.total:
                self.emit(self._mock_image(step), step + 1)
            return
        try:
            self._real(step, latents)
        except Exception:
            log.exception("Preview failed; disabling previews for this job")
            self.failed = True

    def _real(self, step, latents):
        import torch

        x = latents[:1].detach().float()
        prev, self.prev = self.prev, x
        last = step + 1 == self.total
        if prev is None or not ((step + 1) % self.every == 0 or last):
            return
        sigmas = self.pipe.scheduler.sigmas
        s0, s1 = float(sigmas[step]), float(sigmas[step + 1])
        x0 = x if s1 == s0 else x - s1 * (x - prev) / (s1 - s0)
        gh, gw = self.grid
        if x0.shape[1] != gh * gw:
            return  # unexpected layout (e.g. a pipeline change); skip rather than draw garbage
        with torch.no_grad():
            img = self._fast(x0, gh, gw) if self.mode == "fast" else self._full(x0)
        self.emit(img, step + 1)

    def _fast(self, x0, gh, gw):
        import torch

        weights = _fit_fast_map(self.pipe).to(x0.device)
        feats = torch.cat([x0[0], torch.ones(x0.shape[1], 1, device=x0.device)], dim=1)
        rgba = (feats @ weights).clamp(0, 1).reshape(gh, gw, 4)
        arr = (rgba * 255).round().to(torch.uint8).cpu().numpy()
        img = Image.fromarray(arr, "RGBA")
        scale = FAST_MAX_SIDE / max(gw, gh)
        return img.resize((max(1, round(gw * scale)), max(1, round(gh * scale))), Image.Resampling.BILINEAR)

    def _full(self, x0):
        import torch

        pipe = self.pipe
        vae = pipe.vae
        lat = pipe._unpack_latents(x0, self.height, self.width, pipe.vae_scale_factor).to(vae.dtype)
        mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        std = torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lat.device, lat.dtype)
        lat = lat * std + mean
        decoded = vae.decode(lat.to(pipe._execution_device), return_dict=False)[0][:, :, 0]
        return pipe.image_processor.postprocess(decoded, output_type="pil")[0]

    def _mock_image(self, step):
        frac = (step + 1) / self.total
        noise = Image.effect_noise((self.width // 4, self.height // 4), 80).convert("RGBA")
        tint = Image.new("RGBA", noise.size, (90, 120, 200, 255))
        return Image.blend(noise, tint, frac)
