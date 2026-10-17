"""Selection masks: building them from user input and using them to guide/constrain the model.

A selection is always normalised to a single-channel ``L`` image the same size as the source
image, where 255 means "selected". It can come from a painted mask image, from one or more
lasso polygons, or from rectangles.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw, ImageFilter

Point = tuple[float, float]

ANNOTATION_COLOR = (255, 0, 0, 255)


def mask_from_image(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Turn an uploaded mask (any mode) into an ``L`` mask of ``size``.

    Images with a meaningful alpha channel use alpha as the selection (painted strokes on a
    transparent layer); everything else uses luminance (white = selected).
    """
    if mask.mode in ("RGBA", "LA", "PA") or (mask.mode == "P" and "transparency" in mask.info):
        rgba = mask.convert("RGBA")
        alpha = rgba.getchannel("A")
        lo, hi = alpha.getextrema()
        m = alpha if lo < 255 else rgba.convert("L")
    else:
        m = mask.convert("L")
    if m.size != size:
        m = m.resize(size, Image.Resampling.NEAREST)
    return m.point(lambda v: 255 if v >= 128 else 0)


def mask_from_polygons(polygons: Iterable[Sequence[Point]], size: tuple[int, int]) -> Image.Image:
    m = Image.new("L", size, 0)
    draw = ImageDraw.Draw(m)
    for poly in polygons:
        pts = [(float(x), float(y)) for x, y in poly]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return m


def mask_from_boxes(boxes: Iterable[Sequence[float]], size: tuple[int, int]) -> Image.Image:
    m = Image.new("L", size, 0)
    draw = ImageDraw.Draw(m)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)], fill=255)
    return m


def parse_selection_json(text: str) -> dict:
    """Parse the ``selection`` field accepted by the API.

    Accepted shapes: ``{"polygons": [[[x, y], ...], ...], "boxes": [[x0, y0, x1, y1], ...],
    "points": [[x, y], ...]}``, or a bare list of polygons. Each of ``points`` auto-selects the
    object under it (see :mod:`qwen_image.segment`).
    """
    data = json.loads(text)
    if isinstance(data, list):
        data = {"polygons": data}
    if not isinstance(data, dict):
        raise ValueError("selection must be a JSON object or a list of polygons")
    points = data.get("points") or []
    if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in points):
        raise ValueError("points must be a list of [x, y] pairs")
    return {"polygons": data.get("polygons") or [], "boxes": data.get("boxes") or [], "points": points}


def build_mask(
    size: tuple[int, int],
    mask: Image.Image | None = None,
    polygons: Iterable[Sequence[Point]] | None = None,
    boxes: Iterable[Sequence[float]] | None = None,
) -> Image.Image | None:
    """Union of every selection source given; ``None`` when nothing was given."""
    parts = []
    if mask is not None:
        parts.append(mask_from_image(mask, size))
    polygons = list(polygons or [])
    if polygons:
        parts.append(mask_from_polygons(polygons, size))
    boxes = list(boxes or [])
    if boxes:
        parts.append(mask_from_boxes(boxes, size))
    if not parts:
        return None
    out = parts[0]
    for p in parts[1:]:
        out = ImageChops.lighter(out, p)
    return out


def is_empty(mask: Image.Image | None) -> bool:
    return mask is None or mask.getbbox() is None


def _odd(n: float) -> int:
    n = max(3, int(round(n)))
    return n if n % 2 else n + 1


def dilate(mask: Image.Image, px: int) -> Image.Image:
    if px <= 0:
        return mask
    # MaxFilter cost grows with kernel size, so reach large radii with repeated small passes.
    out = mask
    remaining = px
    while remaining > 0:
        step = min(remaining, 7)
        out = out.filter(ImageFilter.MaxFilter(_odd(step * 2 + 1)))
        remaining -= step
    return out


def feather(mask: Image.Image, px: float) -> Image.Image:
    return mask.filter(ImageFilter.GaussianBlur(px)) if px > 0 else mask


def outline(mask: Image.Image, width: int) -> Image.Image:
    """A band of ``width`` pixels hugging the outside edge of the selection."""
    return ImageChops.subtract(dilate(mask, width), mask)


def outline_width(size: tuple[int, int]) -> int:
    return max(3, round(max(size) / 200))


def annotate(image: Image.Image, mask: Image.Image, color=ANNOTATION_COLOR) -> Image.Image:
    """Draw a solid outline around the selection, the way a user would circle something.

    Qwen-Image-2.1 is trained to take local-edit instructions from circled/painted regions, so
    an outlined copy of the image is how a selection is communicated to the model.
    """
    base = image.convert("RGBA")
    band = outline(mask, outline_width(image.size))
    layer = Image.new("RGBA", base.size, color)
    return Image.composite(layer, base, band)


def padded_bbox(mask: Image.Image, pad_frac: float = 0.12, min_pad: int = 16) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = mask.getbbox() or (0, 0, *mask.size)
    pad = max(min_pad, int(max(x1 - x0, y1 - y0) * pad_frac))
    w, h = mask.size
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def apply_alpha(image: Image.Image, alpha: Image.Image) -> Image.Image:
    """Multiply ``image``'s existing alpha by ``alpha``."""
    rgba = image.convert("RGBA")
    if alpha.size != rgba.size:
        alpha = alpha.resize(rgba.size, Image.Resampling.BILINEAR)
    rgba.putalpha(ImageChops.multiply(rgba.getchannel("A"), alpha))
    return rgba


def composite_inside(original: Image.Image, edited: Image.Image, mask: Image.Image, feather_px: float) -> Image.Image:
    """Keep ``edited`` only inside the (feathered) selection and ``original`` everywhere else."""
    edited = edited.resize(original.size, Image.Resampling.LANCZOS)
    mode = "RGBA" if "A" in original.getbands() or "A" in edited.getbands() else "RGB"
    soft = feather(dilate(mask, int(feather_px)), feather_px)
    return Image.composite(edited.convert(mode), original.convert(mode), soft)


def has_transparency(image: Image.Image, threshold: int = 250) -> bool:
    if "A" not in image.getbands():
        return False
    return image.getchannel("A").getextrema()[0] < threshold
