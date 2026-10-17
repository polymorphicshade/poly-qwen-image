"""Auto selection: click on something and get a mask of the object under the click.

Qwen-Image-2.1 is a generator and has no point-prompted segmentation, so this uses Meta's
SAM 2.1 (``facebook/sam2.1-hiera-large``, Apache-2.0, ~0.9 GB) through transformers. It
segments by context, not only by colour: a click on a shirt selects the shirt, and further
clicks refine the selection (positive points add to it, negative points cut away from it).

Running the image encoder is the expensive part, so its embeddings are cached per image and
refinement clicks only run the light prompt decoder (~0.1 s on a GPU, ~0.05 s on a CPU).

There is also a ``color`` method, a classic magic wand (flood fill of similar colours) that
needs no model.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence

import numpy as np
from PIL import Image, ImageChops

log = logging.getLogger(__name__)

SEGMENT_REPO = os.environ.get("QWEN_IMAGE_SEGMENT_MODEL", "facebook/sam2.1-hiera-large")
SEGMENT_GB = 0.9
METHODS = ("ai", "color")
# Below this much free VRAM (e.g. while a generation runs) the model runs on the CPU instead.
_MIN_FREE_VRAM = 3 * 1024**3


def image_key(image: Image.Image) -> str:
    h = hashlib.sha1(f"{image.mode}{image.size}".encode())
    h.update(image.tobytes())
    return h.hexdigest()[:16]


def is_downloaded(repo: str = SEGMENT_REPO) -> bool:
    if repo == "mock":
        return True
    from huggingface_hub import snapshot_download

    try:
        snapshot_download(repo, local_files_only=True)
    except Exception:
        return False
    return True


def download(report: Callable[[str], None] | None = None, repo: str = SEGMENT_REPO) -> str:
    from huggingface_hub import snapshot_download

    report = report or (lambda msg: log.info(msg))
    report(f"Downloading {repo} (cached files are reused)")
    path = snapshot_download(repo)
    report("Download complete")
    return path


def flood_select(image: Image.Image, point: Sequence[float], tolerance: int = 32) -> Image.Image:
    """Magic wand: the connected region of colours within ``tolerance`` of the clicked pixel."""
    import cv2

    rgb = np.array(image.convert("RGB"))
    h, w = rgb.shape[:2]
    x, y = min(w - 1, max(0, int(point[0]))), min(h - 1, max(0, int(point[1])))
    flood = np.zeros((h + 2, w + 2), np.uint8)
    t = (int(tolerance),) * 3
    flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    cv2.floodFill(rgb, flood, (x, y), (0, 0, 0), t, t, flags)
    return Image.fromarray(flood[1:-1, 1:-1])


class Segmenter:
    """Point/box-prompted segmentation with per-image embedding caching. Thread-safe."""

    def __init__(self, repo: str = SEGMENT_REPO, device: str | None = None, cache_size: int = 6):
        self.repo = repo
        self.device = device or os.environ.get("QWEN_IMAGE_SEGMENT_DEVICE", "auto")
        self.cache_size = cache_size
        self._model = self._processor = None
        self._lock = threading.RLock()
        self._images: OrderedDict[str, Image.Image] = OrderedDict()
        self._embeddings: OrderedDict[str, list] = OrderedDict()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------ image cache

    def remember(self, image: Image.Image) -> str:
        """Keep ``image`` so later calls can refer to it by the returned id."""
        key = image_key(image)
        with self._lock:
            self._images[key] = image
            self._images.move_to_end(key)
            while len(self._images) > self.cache_size:
                old, _ = self._images.popitem(last=False)
                self._embeddings.pop(old, None)
        return key

    def image(self, key: str) -> Image.Image | None:
        with self._lock:
            return self._images.get(key)

    # ------------------------------------------------------------------ model

    def _load(self, report: Callable[[str], None]):
        if self._model is None:
            from transformers import Sam2Model, Sam2Processor

            report(f"Loading auto-select model {self.repo}")
            self._processor = Sam2Processor.from_pretrained(self.repo)
            self._model = Sam2Model.from_pretrained(self.repo).eval()
        return self._model, self._processor

    def _pick_device(self) -> str:
        import torch

        if self.device != "auto":
            return self.device
        if not torch.cuda.is_available():
            return "cpu"
        if next(self._model.parameters()).is_cuda:
            return "cuda"
        free, _ = torch.cuda.mem_get_info()
        return "cuda" if free >= _MIN_FREE_VRAM else "cpu"

    def release_gpu(self) -> None:
        """Move the model off the GPU, e.g. before a generation needs the VRAM."""
        with self._lock:
            if self._model is None:
                return
            import torch

            if next(self._model.parameters()).is_cuda:
                self._model.to("cpu")
                torch.cuda.empty_cache()

    # ------------------------------------------------------------------ segmentation

    def segment(
        self,
        image: Image.Image,
        points: Sequence[Sequence[float]] = (),
        labels: Sequence[int] | None = None,
        box: Sequence[float] | None = None,
        method: str = "ai",
        tolerance: int = 32,
        report: Callable[[str], None] | None = None,
    ) -> tuple[Image.Image, dict]:
        """Mask (``L``, 255 = selected) of the one object the prompts point at.

        ``points`` are ``[x, y]`` in image pixels; ``labels`` mark each as part of the object (1)
        or not (0), and default to all 1. ``box`` optionally bounds the object.
        """
        points = [[float(p[0]), float(p[1])] for p in points]
        labels = [int(v) for v in labels] if labels is not None else [1] * len(points)
        if len(labels) != len(points):
            raise ValueError("labels must have one entry per point")
        if any(v not in (0, 1) for v in labels):
            raise ValueError("labels must be 1 (part of the object) or 0 (not part of it)")
        if box is not None and len(box) != 4:
            raise ValueError("box must be [x0, y0, x1, y1]")
        if not any(labels) and box is None:
            raise ValueError("Give at least one positive point or a box")
        if method not in METHODS:
            raise ValueError(f"method must be one of {', '.join(METHODS)}")
        if method == "color" or self.repo == "mock":
            return self._color(image, points, labels, box, tolerance)
        return self._sam(image, points, labels, box, report or (lambda msg: None))

    def select_at(self, image: Image.Image, points: Sequence[Sequence[float]], method: str = "ai") -> Image.Image:
        """Union of the objects under each point (each point is its own object)."""
        out = Image.new("L", image.size, 0)
        for p in points:
            mask, _ = self.segment(image, [p], method=method)
            out = ImageChops.lighter(out, mask)
        return out

    def _color(self, image, points, labels, box, tolerance) -> tuple[Image.Image, dict]:
        out = Image.new("L", image.size, 0)
        for p, label in zip(points, labels):
            if label:
                out = ImageChops.lighter(out, flood_select(image, p, tolerance))
        for p, label in zip(points, labels):
            if not label:
                out = ImageChops.subtract(out, flood_select(image, p, tolerance))
        if box is not None:
            from .selection import mask_from_boxes

            region = mask_from_boxes([box], image.size)
            out = ImageChops.multiply(out, region) if points else region
        return out, {"method": "color", "score": None}

    def _sam(self, image, points, labels, box, report) -> tuple[Image.Image, dict]:
        import torch

        rgb = image.convert("RGB")
        key = self.remember(image)
        with self._lock, torch.inference_mode():
            model, proc = self._load(report)
            device = self._pick_device()
            model.to(device)
            emb = self._embeddings.get(key)
            if emb is None:
                pixels = proc(images=rgb, return_tensors="pt")["pixel_values"].to(device)
                emb = [e.cpu() for e in model.get_image_embeddings(pixels)]
                self._embeddings[key] = emb
            sizes = [[rgb.height, rgb.width]]
            kwargs: dict = {"original_sizes": sizes, "return_tensors": "pt"}
            if points:
                kwargs |= {"input_points": [[points]], "input_labels": [[labels]]}
            if box is not None:
                x0, y0, x1, y1 = box
                kwargs["input_boxes"] = [[[min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]]]
            prompts = proc(**kwargs)
            # One point is ambiguous (a button, the shirt, the person?): take SAM's best guess of
            # three. With several prompts the object is well defined, and one mask is better.
            multi = len(points) == 1 and box is None
            out = model(
                image_embeddings=[e.to(device) for e in emb],
                input_points=prompts["input_points"].to(device) if points else None,
                input_labels=prompts["input_labels"].to(device) if points else None,
                input_boxes=prompts["input_boxes"].to(device) if box is not None else None,
                multimask_output=multi,
            )
            scores = out.iou_scores[0, 0].float().cpu()
            best = int(scores.argmax())
            masks = proc.post_process_masks(out.pred_masks.cpu(), sizes)[0][0]
        mask = Image.fromarray(masks[best].numpy().astype(np.uint8) * 255)
        return mask, {"method": "ai", "score": round(float(scores[best]), 4), "device": device}
