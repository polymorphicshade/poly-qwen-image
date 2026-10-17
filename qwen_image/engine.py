"""The core operations: generate, edit, combine and isolate.

``Engine`` owns at most one loaded pipeline and is not thread-safe; the API serialises access
through the job queue, and the CLI uses it directly.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, fields

from PIL import Image, ImageChops, ImageOps

from . import selection as sel
from .preview import Previewer, output_dims
from .variants import Variant, get_variant

log = logging.getLogger(__name__)

MAX_REFERENCES = 10


class Cancelled(Exception):
    pass


@dataclass
class Options:
    """Sampling options shared by every operation."""

    width: int | None = None
    height: int | None = None
    steps: int | None = None
    seed: int | None = None
    cfg: float = 1.0
    negative_prompt: str | None = None
    num_images: int = 1
    resolution: int = 1024
    transparent: bool = False
    # Live previews while sampling: "off", "fast" (approximate, every step) or "full" (VAE decode).
    preview: str = "off"
    preview_every: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> Options:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names and v is not None})


@dataclass
class Hooks:
    progress: Callable[[int, int], None] = lambda step, total: None
    status: Callable[[str], None] = lambda msg: None
    cancelled: Callable[[], bool] = lambda: False
    # Receives (preview image, step) when Options.preview is enabled.
    preview: Callable[[Image.Image, int], None] = lambda image, step: None


@dataclass
class Result:
    images: list[Image.Image]
    seed: int
    prompt: str
    variant: str
    info: dict = field(default_factory=dict)
    # Additional named images, e.g. the control map a ControlNet run used.
    extras: dict = field(default_factory=dict)


def transparent_prompt(prompt: str) -> str:
    """The prompt wrapper Qwen recommends for native RGBA output."""
    body = prompt.strip().rstrip(".")
    return f"This is an RGBA image with transparency. {body}. The image has alpha channel and the background is transparent."


def _rgb_if_opaque(img: Image.Image) -> Image.Image:
    return img.convert("RGB") if "A" in img.getbands() and not sel.has_transparency(img) else img


class Engine:
    def __init__(self, variant: str | None = None, offload: str = "auto"):
        self.offload = offload
        self.default_variant = variant
        self.variant: Variant | None = None
        self.pipe = None
        self._lock = threading.RLock()
        self.control_branch = None
        from .annotators import Annotators
        from .segment import SEGMENT_REPO, Segmenter

        self.annotators = Annotators()
        # Auto selection (click-to-select). The mock variant gets a model-free stand-in.
        self.segmenter = Segmenter("mock" if variant == "mock" else SEGMENT_REPO)

    # ------------------------------------------------------------------ model management

    def load(self, variant: str | None = None, hooks: Hooks | None = None) -> Variant:
        from .models import build_pipeline

        hooks = hooks or Hooks()
        v = get_variant(variant or self.default_variant)
        # Sampling needs the VRAM more than an idle auto-select model does.
        self.segmenter.release_gpu()
        with self._lock:
            if self.variant and self.variant.name == v.name and self.pipe is not None:
                return v
            self.unload()
            hooks.status(f"Loading {v.label}")
            t0 = time.time()
            self.pipe = build_pipeline(v, self.offload, hooks.status)
            self.variant = v
            hooks.status(f"Loaded {v.label} in {time.time() - t0:.0f}s")
            return v

    def unload(self) -> None:
        with self._lock:
            if self.pipe is None:
                return
            self.pipe = None
            self.variant = None
            import gc

            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    # ------------------------------------------------------------------ sampling

    def _run(
        self,
        prompt: str,
        images: list[Image.Image] | None,
        opts: Options,
        hooks: Hooks,
        variant: str | None,
        pipe_kwargs: dict | None = None,
    ) -> tuple[list[Image.Image], int]:
        v = self.load(variant, hooks)
        pipe = self.pipe
        seed = opts.seed if opts.seed is not None and opts.seed >= 0 else random.randrange(2**31)

        if v.name == "mock":
            import types

            generator = types.SimpleNamespace(initial_seed=lambda: seed)
        else:
            import torch

            generator = torch.Generator("cpu").manual_seed(seed)

        # Turbo checkpoints ship a fixed sigma grid that decides the step count.
        steps = None if v.turbo else opts.steps
        total = 8 if v.turbo else (steps or 40)

        # Resolve the output size up front (as the pipeline would) so previews know the latent grid.
        width, height = output_dims(opts.width, opts.height, images, opts.resolution)
        previewer = None
        if opts.preview and opts.preview != "off":
            if opts.preview == "fast":
                hooks.status("Preparing previews")
            previewer = Previewer(pipe, opts.preview, opts.preview_every, width, height, total, hooks.preview, mock=v.name == "mock")

        def on_step(p, i, t, kwargs):
            hooks.progress(i + 1, total)
            if previewer is not None:
                previewer(i, kwargs.get("latents"))
            if hooks.cancelled():
                p._interrupt = True
            return kwargs

        kwargs = dict(
            prompt=prompt,
            image=images or None,
            width=width,
            height=height,
            generator=generator,
            num_images_per_prompt=max(1, opts.num_images),
            output_resolution=opts.resolution,
            callback_on_step_end=on_step,
        )
        if steps:
            kwargs["num_inference_steps"] = steps
        kwargs.update(pipe_kwargs or {})
        if opts.cfg and opts.cfg > 1 and not v.turbo:
            kwargs["true_cfg_scale"] = opts.cfg
            kwargs["negative_prompt"] = opts.negative_prompt or " "

        hooks.status("Sampling")
        with self._lock:
            out = pipe(**kwargs)
        if hooks.cancelled():
            raise Cancelled()
        return list(out.images), seed

    def _result(self, images, seed, prompt, extras=None, **info) -> Result:
        variant = self.variant.name if self.variant else ""
        return Result(images=images, seed=seed, prompt=prompt, variant=variant, info=info, extras=extras or {})

    # ------------------------------------------------------------------ operations

    def generate(
        self,
        prompt: str,
        references: list[Image.Image] | None = None,
        opts: Options | None = None,
        hooks: Hooks | None = None,
        variant: str | None = None,
    ) -> Result:
        """Text-to-image, optionally conditioned on reference images (style, subject, layout)."""
        opts, hooks = opts or Options(), hooks or Hooks()
        references = list(references or [])
        if len(references) > MAX_REFERENCES:
            raise ValueError(f"At most {MAX_REFERENCES} reference images are supported")
        if not prompt.strip():
            raise ValueError("A prompt is required")
        full = transparent_prompt(prompt) if opts.transparent else prompt
        imgs, seed = self._run(full, references, opts, hooks, variant)
        if not opts.transparent:
            imgs = [_rgb_if_opaque(i) for i in imgs]
        return self._result(imgs, seed, full, references=len(references))

    def edit(
        self,
        image: Image.Image,
        prompt: str,
        mask: Image.Image | None = None,
        mask_mode: str = "annotate",
        keep_outside: bool = True,
        feather_px: float = 8.0,
        opts: Options | None = None,
        hooks: Hooks | None = None,
        variant: str | None = None,
    ) -> Result:
        """Instruction-based editing; an optional selection mask restricts the edit to a region.

        ``mask_mode`` picks how the region is shown to the model: ``annotate`` circles it in red on
        the input, ``reference`` passes a black/white mask as a second image. With
        ``keep_outside`` the output is composited back so pixels outside the selection are the
        original ones.
        """
        opts, hooks = opts or Options(), hooks or Hooks()
        if not prompt.strip():
            raise ValueError("A prompt is required")
        has_mask = not sel.is_empty(mask)
        body = prompt.strip().rstrip(".")
        if has_mask and mask_mode == "annotate":
            inputs = [sel.annotate(image, mask)]
            full = (
                f"{body}. Apply this change only inside the area outlined in red and remove the red outline. "
                f"Keep everything outside the outlined area exactly unchanged."
            )
        elif has_mask and mask_mode == "reference":
            inputs = [image, mask.convert("RGB")]
            full = (
                f"Edit the first image: {body}. Only change the region that is white in the second image (a mask); "
                f"keep everything else in the first image exactly unchanged. Output the edited first image."
            )
        elif has_mask:
            raise ValueError("mask_mode must be 'annotate' or 'reference'")
        else:
            inputs, full = [image], prompt
        if opts.transparent:
            full = transparent_prompt(full)
        # Without an explicit size the pipeline follows the last condition image's aspect ratio,
        # which is the input (or its same-sized mask), and the result is scaled back to the input size.
        imgs, seed = self._run(full, inputs, opts, hooks, variant)
        out = []
        for img in imgs:
            if has_mask and keep_outside:
                img = sel.composite_inside(image, img, mask, feather_px)
            elif opts.width is None and opts.height is None:
                img = img.resize(image.size, Image.Resampling.LANCZOS)
            out.append(img if opts.transparent else _rgb_if_opaque(img))
        return self._result(out, seed, full, masked=has_mask, mask_mode=mask_mode if has_mask else None)

    # ------------------------------------------------------------------ ControlNet

    def _control_branch(self, hooks: Hooks):
        """The ControlNet-Union branch, loaded once and kept in CPU RAM between runs."""
        from . import controlnet as cn

        if self.control_branch is None:
            self.control_branch = cn.load_branch(report=hooks.status)
        return self.control_branch

    def detect_control(
        self,
        control_type: str,
        image: Image.Image,
        options: dict | None = None,
        hooks: Hooks | None = None,
    ) -> Result:
        """Extract a control map (and, for pose, editable keypoints) from ``image``."""
        hooks = hooks or Hooks()
        hooks.status(f"Extracting {control_type} map")
        with self._lock:
            cmap, info = self.annotators.detect(control_type, image, **(options or {}))
        return Result(images=[cmap], seed=-1, prompt="", variant="", info={"control_type": control_type, **info})

    def controlnet(
        self,
        prompt: str,
        control_type: str = "pose",
        control_map: Image.Image | None = None,
        source: Image.Image | None = None,
        pose: dict | None = None,
        references: list[Image.Image] | None = None,
        inpaint_image: Image.Image | None = None,
        mask: Image.Image | None = None,
        strength: float | None = None,
        keep_outside: bool = True,
        feather_px: float = 8.0,
        invert: bool = False,
        detect_options: dict | None = None,
        opts: Options | None = None,
        hooks: Hooks | None = None,
        variant: str | None = None,
    ) -> Result:
        """Generate guided by a control signal (Qwen-Image-2.1 Fun ControlNet-Union).

        The control signal is, in order of precedence: ``pose`` keypoints (rendered as a skeleton),
        a ready-made ``control_map``, or a ``source`` image that is run through the ``control_type``
        preprocessor. ``references`` are optional subject images whose look is kept (e.g. re-posing
        a character); with them, ``strength`` defaults to 0.8 so the reference can still shape the
        result. ``inpaint_image`` + ``mask`` repaint only the masked region (white = repaint), and
        can be combined with a control signal so the repainted part also follows it.
        """
        from . import annotators as ann
        from . import controlnet as cn

        opts, hooks = opts or Options(), hooks or Hooks()
        references = list(references or [])
        if not prompt.strip():
            raise ValueError("A prompt describing the image is required")
        if control_type not in ann.CONTROL_TYPES:
            raise ValueError(f"control_type must be one of {', '.join(ann.CONTROL_TYPES)}")
        if len(references) > MAX_REFERENCES:
            raise ValueError(f"At most {MAX_REFERENCES} reference images are supported")
        has_mask = not sel.is_empty(mask)
        if inpaint_image is not None and not has_mask:
            raise ValueError("Inpainting needs a mask over the region to repaint")
        if pose is None and control_map is None and source is None and inpaint_image is None:
            raise ValueError("Provide a control map, a source image to extract one from, pose keypoints, or an image to inpaint")

        # Output size: explicit, else the aspect of whatever defines the layout.
        if opts.width or opts.height:
            width, height = output_dims(opts.width, opts.height, None, opts.resolution)
        else:
            basis = inpaint_image or control_map or source
            if basis is None and pose is not None:
                basis = Image.new("L", (int(pose.get("width") or 768), int(pose.get("height") or 1024)))
            width, height = output_dims(None, None, [basis] if basis is not None else None, opts.resolution)

        if pose is not None:
            cmap = ann.render_pose(ann.validate_pose(pose), width, height)
        elif control_map is not None:
            cmap = ann.prepare_map(control_map, invert).resize((width, height), Image.Resampling.BILINEAR)
        elif source is not None:
            hooks.status(f"Extracting {control_type} map")
            with self._lock:
                cmap, _ = self.annotators.detect(control_type, source, **(detect_options or {}))
            cmap = cmap.resize((width, height), Image.Resampling.BILINEAR)
        else:
            cmap = None  # pure inpainting

        if strength is None:
            strength = 0.8 if references else 1.0
        full = prompt.strip()
        if references and "<image" not in full:
            tags = ", ".join(f"<image{i + 1}>" for i in range(len(references)))
            subject = "subject" if len(references) == 1 else "subjects"
            full = f"A new image of the same {subject} as in {tags}, keeping their identity, appearance and clothing: {full}"
        if opts.transparent:
            full = transparent_prompt(full)

        v = self.load(variant, hooks)
        run_opts = Options(**{**opts.__dict__, "width": width, "height": height})
        inpaint = inpaint_image.resize((width, height), Image.Resampling.LANCZOS) if inpaint_image is not None else None
        region = mask.resize((width, height), Image.Resampling.NEAREST) if has_mask else None
        if v.name == "mock":
            imgs, seed = self._run(full, references, run_opts, hooks, variant)
        else:
            branch = self._control_branch(hooks)
            hooks.status("Encoding control signal")
            ctx = cn.build_context(self.pipe, width, height, control_image=cmap, inpaint_image=inpaint, mask=region)
            with cn.attached(self.pipe.transformer, branch, ctx, float(strength)):
                imgs, seed = self._run(full, references, run_opts, hooks, variant, pipe_kwargs={"use_kv_cache": False})

        out = []
        for img in imgs:
            if inpaint is not None and keep_outside:
                img = sel.composite_inside(inpaint, img, region, feather_px)
            out.append(img if opts.transparent else _rgb_if_opaque(img))
        extras = {"control": cmap} if cmap is not None else {}
        return self._result(
            out, seed, full, control_type=control_type, strength=strength, references=len(references),
            inpaint=inpaint is not None, size=[width, height], extras=extras,
        )

    def face(
        self,
        image: Image.Image,
        face_ref: Image.Image,
        mask: Image.Image | None = None,
        prompt: str = "",
        keep_hair: bool = True,
        keep_expression: bool = True,
        keep_outside: bool = True,
        feather_px: float = 6.0,
        opts: Options | None = None,
        hooks: Hooks | None = None,
        variant: str | None = None,
    ) -> Result:
        """Give the subject of ``image`` the face of the person/character in ``face_ref``.

        Meant for keeping a character consistent across images. ``face_ref`` and the target go
        to the model as two condition images (target last, so the output keeps its framing).
        An optional ``mask`` circles the face to change - useful with several people - and with
        ``keep_outside`` everything outside it keeps the original pixels.
        """
        opts, hooks = opts or Options(), hooks or Hooks()
        has_mask = not sel.is_empty(mask)
        identity = "facial structure, eyes, eyebrows, nose, mouth, jawline, skin tone and distinguishing marks"
        keep = ["head angle", "lighting", "body", "clothing", "background", "art style"]
        if keep_hair:
            keep.insert(1, "hairstyle and hair colour")
        else:
            identity += ", hairstyle and hair colour"
        if keep_expression:
            keep.insert(0, "facial expression")
        else:
            identity += ", and facial expression"
        target = "the person outlined in red in the second image" if has_mask else "the person in the second image"
        parts = [
            f"Replace the face of {target} with the face of the person in the first image, so that they are clearly "
            f"the same person: same {identity}.",
            f"Keep the second image's {', '.join(keep[:-1])} and {keep[-1]} unchanged.",
        ]
        if has_mask:
            parts.append("Only change the outlined face and remove the red outline.")
        if prompt.strip():
            parts.append(prompt.strip().rstrip(".") + ".")
        full = " ".join(parts)
        if opts.transparent:
            full = transparent_prompt(full)
        target_img = sel.annotate(image, mask) if has_mask else image
        imgs, seed = self._run(full, [face_ref, target_img], opts, hooks, variant)
        out = []
        for img in imgs:
            if has_mask and keep_outside:
                img = sel.composite_inside(image, img, mask, feather_px)
            elif opts.width is None and opts.height is None:
                img = img.resize(image.size, Image.Resampling.LANCZOS)
            out.append(img if opts.transparent else _rgb_if_opaque(img))
        return self._result(out, seed, full, masked=has_mask, keep_hair=keep_hair, keep_expression=keep_expression)

    def combine(
        self,
        images: list[Image.Image],
        prompt: str,
        opts: Options | None = None,
        hooks: Hooks | None = None,
        variant: str | None = None,
    ) -> Result:
        """Compose subjects from several reference images into one new image."""
        opts, hooks = opts or Options(), hooks or Hooks()
        if len(images) < 2:
            raise ValueError("Combine needs at least two images")
        if len(images) > MAX_REFERENCES:
            raise ValueError(f"At most {MAX_REFERENCES} images are supported")
        if not prompt.strip():
            raise ValueError("A prompt is required")
        full = transparent_prompt(prompt) if opts.transparent else prompt
        if opts.width is None and opts.height is None:
            opts = Options(**{**opts.__dict__, "width": opts.resolution, "height": opts.resolution})
        imgs, seed = self._run(full, images, opts, hooks, variant)
        if not opts.transparent:
            imgs = [_rgb_if_opaque(i) for i in imgs]
        return self._result(imgs, seed, full, references=len(images))

    def isolate(
        self,
        image: Image.Image,
        mask: Image.Image,
        description: str = "",
        method: str = "model",
        crop: bool = True,
        keep_canvas: bool = False,
        clip: bool = True,
        restore_pixels: bool = True,
        opts: Options | None = None,
        hooks: Hooks | None = None,
        variant: str | None = None,
    ) -> Result:
        """Cut the selected subject out of ``image`` onto a transparent background.

        ``mask`` is the user's rough outline. With ``method="model"`` the region is circled in red
        and the model extracts the subject as a native RGBA image, which gives a clean alpha
        matte that follows the subject rather than the rough outline; ``method="mask"`` simply
        uses the selection itself as the alpha matte (no model, instant).

        ``crop`` works on a padded crop around the selection, so small sub-elements (an item a
        character holds, an object in the background) fill the model's view. ``clip`` discards
        anything the model returns outside the selection. ``restore_pixels`` keeps the original
        colours and uses only the model's alpha. ``keep_canvas`` returns a full-size image with
        the subject at its original position instead of the crop.
        """
        opts, hooks = opts or Options(), hooks or Hooks()
        if sel.is_empty(mask):
            raise ValueError("Isolate needs a non-empty selection")
        mask = sel.mask_from_image(mask, image.size)
        src = image.convert("RGBA")
        box = sel.padded_bbox(mask) if crop else (0, 0, *image.size)
        work, work_mask = src.crop(box), mask.crop(box)
        # A selection covering everything means "find the subject yourself": nothing to outline.
        whole = work_mask.getextrema()[0] == 255
        seed = -1
        full = ""

        if method == "mask":
            out = sel.apply_alpha(work, sel.feather(work_mask, 1.0))
            outs = [out]
        elif method == "model":
            subject = description.strip() or ("the subject" if not whole else "the main subject")
            where = "" if whole else " outlined in red"
            outline_note = "" if whole else "Remove the red outline and everything else"
            full = transparent_prompt(
                f"Extract {subject}{where} from the input image as a standalone cutout. "
                f"{outline_note or 'Remove everything else'}, including the background and all other objects. "
                f"Keep {subject} exactly as it appears, with the same pose, colours and details"
            )
            o = Options(**{**opts.__dict__, "transparent": True, "width": None, "height": None})
            condition = work if whole else sel.annotate(work, work_mask)
            imgs, seed = self._run(full, [condition], o, hooks, variant)
            # Clip a little outside the outline so the model's edge is kept, not the user's.
            clip_mask = sel.feather(sel.dilate(work_mask, sel.outline_width(work.size) * 2), 2.0)
            outs = []
            for img in imgs:
                img = img.convert("RGBA").resize(work.size, Image.Resampling.LANCZOS)
                if not sel.has_transparency(img):
                    hooks.status("Model returned no transparency; falling back to the selection matte")
                    img = sel.apply_alpha(work, sel.feather(work_mask, 1.0))
                elif restore_pixels:
                    alpha = img.getchannel("A")
                    img = work.copy()
                    img.putalpha(ImageChops.multiply(alpha, work.getchannel("A")))
                if clip:
                    img = sel.apply_alpha(img, clip_mask)
                outs.append(img)
        else:
            raise ValueError("method must be 'model' or 'mask'")

        if keep_canvas and crop:
            placed = []
            for img in outs:
                canvas = Image.new("RGBA", image.size, (0, 0, 0, 0))
                canvas.paste(img, box[:2])
                placed.append(canvas)
            outs = placed
        else:
            outs = [_trim(img) for img in outs]
        return self._result(outs, seed, full, method=method, box=list(box))

    def background(
        self,
        image: Image.Image,
        background: Image.Image,
        mask: Image.Image | None = None,
        prompt: str = "",
        method: str = "model",
        keep_subject: bool = False,
        feather_px: float = 2.0,
        opts: Options | None = None,
        hooks: Hooks | None = None,
        variant: str | None = None,
    ) -> Result:
        """Put the subject of ``image`` in front of the background shown in ``background``.

        ``method="model"`` (default) passes the background reference and the subject image to
        the model, which repaints the scene behind the subject with matching lighting and
        perspective; ``prompt`` adds extra instructions. An optional ``mask`` outlines which
        subject to keep, and ``keep_subject`` pastes the original pixels inside it back (only
        sensible for a tight selection).

        ``method="composite"`` cuts the subject out with an AI matte (as :meth:`isolate`, guided
        by ``mask`` if given, ``prompt`` naming the subject) and lays it over the background image
        resized to cover the frame - nothing is regenerated, but lighting is not adapted.
        """
        opts, hooks = opts or Options(), hooks or Hooks()
        has_mask = not sel.is_empty(mask)

        if method == "composite":
            matte_mask = mask if has_mask else Image.new("L", image.size, 255)
            cut = self.isolate(
                image, matte_mask, prompt, method="model", crop=has_mask, keep_canvas=True, clip=has_mask,
                restore_pixels=True, opts=opts, hooks=hooks, variant=variant,
            )
            plate = ImageOps.fit(background.convert("RGBA"), image.size, Image.Resampling.LANCZOS)
            outs = []
            for subject in cut.images:
                composed = plate.copy()
                composed.alpha_composite(subject.convert("RGBA"))
                outs.append(_rgb_if_opaque(composed))
            return self._result(outs, cut.seed, cut.prompt, method=method, masked=has_mask)
        if method != "model":
            raise ValueError("method must be 'model' or 'composite'")

        subject = "the subject outlined in red in the second image" if has_mask else "the main subject of the second image"
        parts = [
            "Replace the background of the second image with the background from the first image: the same setting, "
            "scenery, colours and atmosphere.",
            f"Keep {subject} exactly unchanged, with the same pose, appearance, size and position in the frame.",
            "Adjust the lighting and shadows on the subject so it sits naturally in the new background.",
        ]
        if has_mask:
            parts.append("Remove the red outline.")
        if prompt.strip():
            parts.append(prompt.strip().rstrip(".") + ".")
        full = " ".join(parts)
        target = sel.annotate(image, mask) if has_mask else image
        imgs, seed = self._run(full, [background, target], opts, hooks, variant)
        outs = []
        for img in imgs:
            if opts.width is None and opts.height is None:
                img = img.resize(image.size, Image.Resampling.LANCZOS)
            if has_mask and keep_subject:
                soft = sel.feather(mask, feather_px)
                mode = "RGBA" if "A" in img.getbands() else "RGB"
                img = Image.composite(image.convert(mode), img.convert(mode), soft.resize(img.size))
            outs.append(_rgb_if_opaque(img))
        return self._result(outs, seed, full, method=method, masked=has_mask, keep_subject=keep_subject)


def _trim(img: Image.Image, pad: int = 4) -> Image.Image:
    """Crop transparent margins."""
    bbox = img.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    if not bbox:
        return img
    x0, y0, x1, y1 = bbox
    return img.crop((max(0, x0 - pad), max(0, y0 - pad), min(img.width, x1 + pad), min(img.height, y1 + pad)))
