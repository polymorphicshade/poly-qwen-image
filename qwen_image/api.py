"""HTTP API (and the static web UI that uses it).

Every operation is a job: ``POST`` returns the job immediately (``202``) and clients poll
``GET /api/jobs/{id}``; pass ``wait=true`` to block until it finishes instead. Images are sent
as multipart file fields so the API is easy to drive with ``curl -F``.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageChops, ImageOps, UnidentifiedImageError

from . import __version__
from . import selection as sel
from .engine import Engine, Options
from .jobs import TERMINAL, Job, JobManager
from .preview import MODES as PREVIEW_MODES
from .variants import DEFAULT_VARIANT, get_variant, list_variants

log = logging.getLogger(__name__)
WEB_DIR = Path(__file__).parent / "web"


def _read_image(upload: UploadFile, field: str) -> Image.Image:
    data = upload.file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"'{field}' is not a readable image: {e}") from e
    img = ImageOps.exif_transpose(img)
    return img if img.mode in ("RGB", "RGBA") else img.convert("RGBA" if "A" in img.getbands() or "transparency" in img.info else "RGB")


def _selection_mask(image: Image.Image, mask: UploadFile | None, selection: str | None, segmenter=None) -> Image.Image | None:
    """The union of an uploaded mask and a ``selection`` JSON, whose ``points`` are auto-selected."""
    mask_img = _read_image(mask, "mask") if mask is not None and mask.filename else None
    polygons, boxes, points = [], [], []
    if selection:
        try:
            parsed = sel.parse_selection_json(selection)
        except ValueError as e:
            raise HTTPException(400, f"Invalid selection: {e}") from e
        polygons, boxes, points = parsed["polygons"], parsed["boxes"], parsed["points"]
    m = sel.build_mask(image.size, mask_img, polygons, boxes)
    if points:
        try:
            auto = segmenter.select_at(image, points)
        except ValueError as e:
            raise HTTPException(400, f"Invalid selection: {e}") from e
        m = auto if m is None else ImageChops.lighter(m, auto)
    return m


def _options(
    width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent, preview="off", preview_every=None
) -> Options:
    if preview not in PREVIEW_MODES:
        raise HTTPException(400, f"preview must be one of {', '.join(PREVIEW_MODES)}")
    if num_images is not None and not 1 <= num_images <= 8:
        raise HTTPException(400, "num_images must be between 1 and 8")
    for name, v in (("width", width), ("height", height)):
        if v is not None and not 256 <= v <= 4096:
            raise HTTPException(400, f"{name} must be between 256 and 4096")
    return Options.from_dict(
        dict(
            width=width,
            height=height,
            steps=steps,
            seed=seed,
            cfg=cfg,
            negative_prompt=negative_prompt or None,
            num_images=num_images,
            resolution=resolution,
            transparent=transparent,
            preview=preview,
            preview_every=preview_every,
        )
    )


def _check_variant(variant: str | None) -> str | None:
    if variant:
        try:
            get_variant(variant)
        except KeyError as e:
            raise HTTPException(400, str(e.args[0])) from e
    return variant or None


def _check_control_type(control_type: str) -> None:
    from .annotators import CONTROL_TYPES

    if control_type not in CONTROL_TYPES:
        raise HTTPException(400, f"control_type must be one of {', '.join(CONTROL_TYPES)}")


def _json_field(text: str | None, name: str) -> dict:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError as e:
        raise HTTPException(400, f"'{name}' is not valid JSON: {e}") from e
    if not isinstance(value, dict):
        raise HTTPException(400, f"'{name}' must be a JSON object")
    return value


def create_app(engine: Engine | None = None, output_dir: str | Path = "outputs") -> FastAPI:
    engine = engine or Engine(DEFAULT_VARIANT)
    jobs = JobManager(engine, output_dir)
    app = FastAPI(title="poly-qwen-image", version=__version__, description="Qwen-Image-2.1 generation, editing, composition and subject isolation.")
    app.state.engine, app.state.jobs = engine, jobs

    async def selection_mask(image, mask, selection):
        # Off the event loop: auto-selected points run the segmentation model.
        return await asyncio.to_thread(_selection_mask, image, mask, selection, engine.segmenter)

    async def respond(job: Job, wait: bool) -> JSONResponse:
        if wait:
            await asyncio.to_thread(job.done_event.wait)
        body = job.to_dict()
        body["queue_position"] = jobs.queue_position(job)
        return JSONResponse(body, status_code=200 if job.status in TERMINAL else 202)

    # Shared multipart fields for sampling options.
    def opts_fields():
        return dict(
            width=Form(None, description="Output width in pixels (default: derived from resolution/aspect)"),
            height=Form(None, description="Output height in pixels (default: derived from resolution/aspect)"),
            steps=Form(None, description="Denoising steps (default 40); more can add detail but take longer. Turbo variants always use 8"),
            seed=Form(None, description="Starting noise; the same seed and settings give the same image. Omit or -1 for random"),
            cfg=Form(1.0, description="True CFG scale: how strictly to follow the prompt. 1 is off; >1 (try 4) enables guidance with negative_prompt and takes about twice as long. Ignored by Turbo variants"),
            negative_prompt=Form(None, description="What to steer away from, e.g. 'blurry, extra fingers'; only used when cfg > 1"),
            num_images=Form(1, description="How many variations to make in one run (1-8)"),
            resolution=Form(1024, description="Output size as the side of a square with the same pixel count, when width/height are not given; the aspect follows the last input image, or is square without one"),
            transparent=Form(False, description="Generate a transparent (RGBA) image"),
            variant=Form(None, description="Model variant to use (default: currently loaded / server default)"),
            wait=Form(False, description="Block until the job finishes"),
            preview=Form("fast", description="Live previews: 'off', 'fast' (approximate) or 'full' (exact VAE decode, slower)"),
            preview_every=Form(None, description="Make a preview every N steps (default 1 for fast, 5 for full)"),
        )

    # ------------------------------------------------------------------ meta

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": __version__, "loaded": engine.variant.name if engine.variant else None}

    @app.get("/api/models")
    async def models():
        from .models import is_downloaded

        async def row(v):
            downloaded = await asyncio.to_thread(is_downloaded, v)
            return v.to_dict() | {"downloaded": downloaded, "loaded": bool(engine.variant and engine.variant.name == v.name)}

        rows = await asyncio.gather(*(row(v) for v in list_variants()))
        from . import controlnet as cn
        from . import segment as seg

        return {
            "segment": {
                "repo": engine.segmenter.repo,
                "approx_download_gb": seg.SEGMENT_GB,
                "downloaded": await asyncio.to_thread(seg.is_downloaded, engine.segmenter.repo),
                "loaded": engine.segmenter.loaded,
            },
            "controlnet": {
                "repo": cn.CONTROLNET_REPO,
                "approx_download_gb": cn.CONTROLNET_GB,
                "downloaded": await asyncio.to_thread(cn.is_downloaded),
                "loaded": engine.control_branch is not None,
            },
            "default": engine.default_variant or DEFAULT_VARIANT,
            "loaded": engine.variant.name if engine.variant else None,
            "variants": rows,
        }

    @app.post("/api/models/load", status_code=202)
    async def load_model(variant: str = Form(...), wait: bool = Form(False)):
        v = _check_variant(variant)

        def run(hooks):
            engine.load(v, hooks)
            engine.default_variant = v

        return await respond(jobs.submit("load", {"variant": v}, run), wait)

    @app.post("/api/models/download", status_code=202)
    async def download_model(variant: str = Form(...), wait: bool = Form(False)):
        from .models import download

        v = get_variant(_check_variant(variant))
        return await respond(jobs.submit("download", {"variant": v.name}, lambda hooks: download(v, hooks.status) and None), wait)

    @app.post("/api/models/unload")
    def unload_model():
        engine.unload()
        return {"ok": True}

    # ------------------------------------------------------------------ operations

    f = opts_fields()

    @app.post("/api/generate", status_code=202, summary="Text-to-image, optionally with reference images")
    async def generate(
        prompt: str = Form(...),
        references: list[UploadFile] = File(default=[], description="Optional reference images (up to 10)"),
        width: int | None = f["width"], height: int | None = f["height"], steps: int | None = f["steps"],
        seed: int | None = f["seed"], cfg: float = f["cfg"], negative_prompt: str | None = f["negative_prompt"],
        num_images: int = f["num_images"], resolution: int = f["resolution"], transparent: bool = f["transparent"],
        variant: str | None = f["variant"], wait: bool = f["wait"],
        preview: str = f["preview"], preview_every: int | None = f["preview_every"],
    ):
        opts = _options(width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent, preview, preview_every)
        refs = [_read_image(r, "references") for r in references if r.filename]
        v = _check_variant(variant)
        params = {"prompt": prompt, "references": len(refs)} | opts.__dict__
        job = jobs.submit("generate", params, lambda hooks: engine.generate(prompt, refs, opts, hooks, v))
        return await respond(job, wait)

    @app.post("/api/edit", status_code=202, summary="Edit an image, optionally within a selected region")
    async def edit(
        image: UploadFile = File(...),
        prompt: str = Form(...),
        mask: UploadFile | None = File(None, description="Selection mask (white/opaque = selected)"),
        selection: str | None = Form(None, description='JSON: {"polygons": [[[x,y],...]], "boxes": [[x0,y0,x1,y1]]}'),
        mask_mode: str = Form("annotate", description="'annotate' (red outline) or 'reference' (mask image)"),
        keep_outside: bool = Form(True, description="Restore original pixels outside the selection"),
        feather: float = Form(8.0, description="Feather radius for keep_outside blending"),
        width: int | None = f["width"], height: int | None = f["height"], steps: int | None = f["steps"],
        seed: int | None = f["seed"], cfg: float = f["cfg"], negative_prompt: str | None = f["negative_prompt"],
        num_images: int = f["num_images"], resolution: int = f["resolution"], transparent: bool = f["transparent"],
        variant: str | None = f["variant"], wait: bool = f["wait"],
        preview: str = f["preview"], preview_every: int | None = f["preview_every"],
    ):
        if mask_mode not in ("annotate", "reference"):
            raise HTTPException(400, "mask_mode must be 'annotate' or 'reference'")
        opts = _options(width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent, preview, preview_every)
        img = _read_image(image, "image")
        m = await selection_mask(img, mask, selection)
        v = _check_variant(variant)
        params = {"prompt": prompt, "mask_mode": mask_mode, "masked": not sel.is_empty(m), "keep_outside": keep_outside} | opts.__dict__
        job = jobs.submit(
            "edit", params,
            lambda hooks: engine.edit(img, prompt, m, mask_mode, keep_outside, feather, opts, hooks, v),
        )
        return await respond(job, wait)

    # ------------------------------------------------------------------ auto selection

    @app.post("/api/segment", summary="Auto-select: the mask of the object at a clicked point")
    async def segment(
        image: UploadFile | None = File(None, description="Image to select in (or pass image_id)"),
        image_id: str | None = Form(None, description="Reuse an image sent before (the X-Image-Id response header) instead of uploading it again"),
        points: str = Form("[]", description="JSON [[x, y], ...] in image pixels"),
        labels: str | None = Form(None, description="JSON [1, 0, ...]: 1 = part of the object, 0 = not part of it (default: all 1)"),
        box: str | None = Form(None, description="JSON [x0, y0, x1, y1]: optional box around the object"),
        method: str = Form("ai", description="'ai' (SAM 2.1: understands objects) or 'color' (magic wand: similar colours)"),
        tolerance: int = Form(32, description="color: maximum per-channel difference from the clicked pixel"),
        output: str = Form("mask", description="'mask' (white on black) or 'alpha' (white on transparent)"),
    ):
        from .segment import METHODS

        if method not in METHODS:
            raise HTTPException(400, f"method must be one of {', '.join(METHODS)}")
        if output not in ("mask", "alpha"):
            raise HTTPException(400, "output must be 'mask' or 'alpha'")
        if image is not None and image.filename:
            img = _read_image(image, "image")
        elif image_id:
            img = engine.segmenter.image(image_id)
            if img is None:
                raise HTTPException(404, "Unknown image_id (it may have expired); upload the image again")
        else:
            raise HTTPException(400, "Provide 'image' or 'image_id'")
        try:
            pts, lbl = json.loads(points or "[]"), json.loads(labels) if labels else None
            bx = json.loads(box) if box else None
        except ValueError as e:
            raise HTTPException(400, f"points, labels and box must be JSON: {e}") from e
        key = engine.segmenter.remember(img)
        try:
            mask, info = await asyncio.to_thread(engine.segmenter.segment, img, pts, lbl, bx, method, tolerance)
        except (ValueError, TypeError, IndexError) as e:
            raise HTTPException(400, str(e)) from e
        if output == "alpha":
            out = Image.new("RGBA", mask.size, (255, 255, 255, 0))
            out.putalpha(mask)
        else:
            out = mask
        buf = io.BytesIO()
        out.save(buf, "PNG")
        bbox = mask.getbbox()
        headers = {
            "X-Image-Id": key,
            "X-Score": "" if info.get("score") is None else str(info["score"]),
            "X-Bbox": ",".join(map(str, bbox)) if bbox else "",
            "Access-Control-Expose-Headers": "X-Image-Id, X-Score, X-Bbox",
        }
        return Response(buf.getvalue(), media_type="image/png", headers=headers)

    @app.post("/api/segment/download", status_code=202, summary="Download the auto-select (SAM 2.1) weights")
    async def segment_download(wait: bool = Form(False)):
        from . import segment as seg

        repo = engine.segmenter.repo
        if repo == "mock":
            raise HTTPException(400, "The mock segmenter has no weights")
        return await respond(jobs.submit("download", {"variant": "segment"}, lambda hooks: seg.download(hooks.status, repo) and None), wait)

    # ------------------------------------------------------------------ ControlNet

    @app.get("/api/controlnet/types", summary="Control types, plus a template pose to start drawing from")
    def controlnet_types():
        from .annotators import CONTROL_TYPES, template_pose

        return {"types": CONTROL_TYPES, "template_pose": template_pose()}

    @app.post("/api/controlnet/download", status_code=202, summary="Download the ControlNet-Union weights")
    async def controlnet_download(wait: bool = Form(False)):
        from . import controlnet as cn

        return await respond(jobs.submit("download", {"variant": "controlnet"}, lambda hooks: cn.download(hooks.status) and None), wait)

    @app.post("/api/controlnet/detect", status_code=202, summary="Extract a control map (and pose keypoints) from an image")
    async def controlnet_detect(
        image: UploadFile = File(..., description="Image to extract the control map from"),
        control_type: str = Form("pose", description="pose, canny, depth, lineart, softedge, mlsd, scribble or grayscale"),
        options: str | None = Form(None, description='JSON detector options, e.g. {"low_threshold": 100, "high_threshold": 200} for canny, {"hands": true, "face": true} for pose'),
        wait: bool = Form(False),
    ):
        _check_control_type(control_type)
        img = _read_image(image, "image")
        opt = _json_field(options, "options")
        job = jobs.submit("detect", {"control_type": control_type, "options": opt}, lambda hooks: engine.detect_control(control_type, img, opt, hooks))
        return await respond(job, wait)

    @app.post("/api/controlnet/pose/render", summary="Render pose keypoints to a control map (PNG)")
    def controlnet_render_pose(body: dict = Body(..., examples=[{"pose": {"width": 768, "height": 1024, "people": []}, "width": 768, "height": 1024}])):
        from .annotators import render_pose, validate_pose

        try:
            pose = validate_pose(body.get("pose"))
            width = int(body.get("width") or pose.get("width") or 768)
            height = int(body.get("height") or pose.get("height") or 1024)
        except (ValueError, TypeError) as e:
            raise HTTPException(400, str(e)) from e
        if not (64 <= width <= 4096 and 64 <= height <= 4096):
            raise HTTPException(400, "width and height must be between 64 and 4096")
        buf = io.BytesIO()
        render_pose(pose, width, height, hands=body.get("hands", True), face=body.get("face", True)).save(buf, "PNG")
        return Response(buf.getvalue(), media_type="image/png")

    @app.post("/api/controlnet", status_code=202, summary="Generate guided by a ControlNet signal (pose, edges, depth, ...)")
    async def controlnet(
        prompt: str = Form(..., description="Describe the whole image to create"),
        control_type: str = Form("pose", description="pose, canny, depth, lineart, softedge, mlsd, scribble or grayscale"),
        control: UploadFile | None = File(None, description="A ready-made control map (white features on black)"),
        source: UploadFile | None = File(None, description="An image to extract the control map from with the control_type preprocessor"),
        pose: str | None = Form(None, description="Pose keypoints JSON (see /api/controlnet/detect); rendered as the control map"),
        references: list[UploadFile] = File(default=[], description="Optional subject images whose look to keep (up to 10)"),
        image: UploadFile | None = File(None, description="Optional image to inpaint (needs mask or selection)"),
        mask: UploadFile | None = File(None, description="Inpaint mask over image: white/opaque = repaint"),
        selection: str | None = Form(None, description='Inpaint region as JSON: {"polygons": [[[x,y],...]], "boxes": [[x0,y0,x1,y1]]}'),
        strength: float | None = Form(None, description="Control strength, 0-1.5 (default 1.0, or 0.8 with references)"),
        invert: bool = Form(False, description="Invert the control map (for black-on-white drawings)"),
        keep_outside: bool = Form(True, description="Inpainting: restore original pixels outside the mask"),
        detect_options: str | None = Form(None, description="JSON options for the preprocessor when 'source' is given"),
        width: int | None = f["width"], height: int | None = f["height"], steps: int | None = f["steps"],
        seed: int | None = f["seed"], cfg: float = f["cfg"], negative_prompt: str | None = f["negative_prompt"],
        num_images: int = f["num_images"], resolution: int = f["resolution"], transparent: bool = f["transparent"],
        variant: str | None = f["variant"], wait: bool = f["wait"],
        preview: str = f["preview"], preview_every: int | None = f["preview_every"],
    ):
        from .annotators import validate_pose

        _check_control_type(control_type)
        if strength is not None and not 0 <= strength <= 2:
            raise HTTPException(400, "strength must be between 0 and 2")
        opts = _options(width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent, preview, preview_every)
        cmap = _read_image(control, "control") if control is not None and control.filename else None
        src = _read_image(source, "source") if source is not None and source.filename else None
        pose_data = None
        if pose:
            try:
                pose_data = validate_pose(json.loads(pose))
            except ValueError as e:
                raise HTTPException(400, f"Invalid pose: {e}") from e
        refs = [_read_image(r, "references") for r in references if r.filename]
        base = _read_image(image, "image") if image is not None and image.filename else None
        region = await selection_mask(base, mask, selection) if base is not None else None
        if base is not None and sel.is_empty(region):
            raise HTTPException(400, "Inpainting needs a mask or selection over the region to repaint")
        if cmap is None and src is None and pose_data is None and base is None:
            raise HTTPException(400, "Provide 'control', 'source', 'pose', or an 'image' + mask to inpaint")
        det = _json_field(detect_options, "detect_options")
        v = _check_variant(variant)
        params = {
            "prompt": prompt, "control_type": control_type, "strength": strength, "references": len(refs),
            "inpaint": base is not None, "control_from": "pose" if pose_data else "map" if cmap else "source" if src else "none",
        } | opts.__dict__
        job = jobs.submit(
            "controlnet", params,
            lambda hooks: engine.controlnet(
                prompt, control_type, cmap, src, pose_data, refs, base, region, strength, keep_outside,
                invert=invert, detect_options=det, opts=opts, hooks=hooks, variant=v,
            ),
        )
        return await respond(job, wait)

    @app.post("/api/face", status_code=202, summary="Give an image's subject the face from a reference image")
    async def face(
        image: UploadFile = File(..., description="Image whose subject's face should change"),
        face: UploadFile = File(..., description="Reference image showing the face to use"),
        mask: UploadFile | None = File(None, description="Optional selection around the face to change"),
        selection: str | None = Form(None, description='JSON: {"polygons": [[[x,y],...]], "boxes": [[x0,y0,x1,y1]]}'),
        prompt: str = Form("", description="Optional extra instructions"),
        keep_hair: bool = Form(True, description="Keep the target's hairstyle (false: take hair from the reference too)"),
        keep_expression: bool = Form(True, description="Keep the target's facial expression"),
        keep_outside: bool = Form(True, description="With a selection, restore original pixels outside it"),
        feather: float = Form(6.0, description="Feather radius for keep_outside blending"),
        width: int | None = f["width"], height: int | None = f["height"], steps: int | None = f["steps"],
        seed: int | None = f["seed"], cfg: float = f["cfg"], negative_prompt: str | None = f["negative_prompt"],
        num_images: int = f["num_images"], resolution: int = f["resolution"], transparent: bool = f["transparent"],
        variant: str | None = f["variant"], wait: bool = f["wait"],
        preview: str = f["preview"], preview_every: int | None = f["preview_every"],
    ):
        opts = _options(width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent, preview, preview_every)
        img = _read_image(image, "image")
        ref = _read_image(face, "face")
        m = await selection_mask(img, mask, selection)
        v = _check_variant(variant)
        params = {
            "prompt": prompt, "keep_hair": keep_hair, "keep_expression": keep_expression,
            "masked": not sel.is_empty(m), "keep_outside": keep_outside,
        } | opts.__dict__
        job = jobs.submit(
            "face", params,
            lambda hooks: engine.face(img, ref, m, prompt, keep_hair, keep_expression, keep_outside, feather, opts, hooks, v),
        )
        return await respond(job, wait)

    @app.post("/api/combine", status_code=202, summary="Combine subjects from 2-10 images into a new image")
    async def combine(
        images: list[UploadFile] = File(..., description="2 to 10 subject images, in the order the prompt refers to them"),
        prompt: str = Form(...),
        width: int | None = f["width"], height: int | None = f["height"], steps: int | None = f["steps"],
        seed: int | None = f["seed"], cfg: float = f["cfg"], negative_prompt: str | None = f["negative_prompt"],
        num_images: int = f["num_images"], resolution: int = f["resolution"], transparent: bool = f["transparent"],
        variant: str | None = f["variant"], wait: bool = f["wait"],
        preview: str = f["preview"], preview_every: int | None = f["preview_every"],
    ):
        opts = _options(width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent, preview, preview_every)
        imgs = [_read_image(i, "images") for i in images if i.filename]
        if not 2 <= len(imgs) <= 10:
            raise HTTPException(400, "combine needs between 2 and 10 images")
        v = _check_variant(variant)
        params = {"prompt": prompt, "images": len(imgs)} | opts.__dict__
        job = jobs.submit("combine", params, lambda hooks: engine.combine(imgs, prompt, opts, hooks, v))
        return await respond(job, wait)

    @app.post("/api/isolate", status_code=202, summary="Isolate a selected subject onto a transparent background")
    async def isolate(
        image: UploadFile = File(...),
        mask: UploadFile | None = File(None, description="Selection mask (white/opaque = selected)"),
        selection: str | None = Form(None, description='JSON: {"polygons": [[[x,y],...]], "boxes": [[x0,y0,x1,y1]]}'),
        description: str = Form("", description="What the selected subject is, e.g. 'the sword the knight holds'"),
        method: str = Form("model", description="'model' (AI matte) or 'mask' (use the selection as-is)"),
        crop: bool = Form(True, description="Work on a crop around the selection (better for small elements)"),
        keep_canvas: bool = Form(False, description="Return a full-size canvas instead of a trimmed cutout"),
        clip: bool = Form(True, description="Discard anything outside the selection"),
        restore_pixels: bool = Form(True, description="Keep original colours; use only the model's alpha"),
        steps: int | None = f["steps"], seed: int | None = f["seed"], cfg: float = f["cfg"],
        negative_prompt: str | None = f["negative_prompt"], num_images: int = f["num_images"],
        resolution: int = f["resolution"], variant: str | None = f["variant"], wait: bool = f["wait"],
        preview: str = f["preview"], preview_every: int | None = f["preview_every"],
    ):
        if method not in ("model", "mask"):
            raise HTTPException(400, "method must be 'model' or 'mask'")
        opts = _options(None, None, steps, seed, cfg, negative_prompt, num_images, resolution, True, preview, preview_every)
        img = _read_image(image, "image")
        m = await selection_mask(img, mask, selection)
        if sel.is_empty(m):
            raise HTTPException(400, "isolate needs a selection: upload 'mask' or pass 'selection'")
        v = _check_variant(variant)
        params = {"description": description, "method": method, "crop": crop, "keep_canvas": keep_canvas, "clip": clip} | opts.__dict__
        job = jobs.submit(
            "isolate", params,
            lambda hooks: engine.isolate(img, m, description, method, crop, keep_canvas, clip, restore_pixels, opts, hooks, v),
        )
        return await respond(job, wait)

    @app.post("/api/background", status_code=202, summary="Replace an image's background with the one from another image")
    async def background(
        image: UploadFile = File(..., description="Image whose subject should be kept"),
        background: UploadFile = File(..., description="Image whose background should be used"),
        mask: UploadFile | None = File(None, description="Optional selection around the subject to keep"),
        selection: str | None = Form(None, description='JSON: {"polygons": [[[x,y],...]], "boxes": [[x0,y0,x1,y1]]}'),
        prompt: str = Form("", description="model: extra instructions; composite: what the subject is"),
        method: str = Form("model", description="'model' (repaint the scene behind the subject) or 'composite' (cut out and paste over)"),
        keep_subject: bool = Form(False, description="model + selection: paste original pixels inside the selection back"),
        width: int | None = f["width"], height: int | None = f["height"], steps: int | None = f["steps"],
        seed: int | None = f["seed"], cfg: float = f["cfg"], negative_prompt: str | None = f["negative_prompt"],
        num_images: int = f["num_images"], resolution: int = f["resolution"],
        variant: str | None = f["variant"], wait: bool = f["wait"],
        preview: str = f["preview"], preview_every: int | None = f["preview_every"],
    ):
        if method not in ("model", "composite"):
            raise HTTPException(400, "method must be 'model' or 'composite'")
        opts = _options(width, height, steps, seed, cfg, negative_prompt, num_images, resolution, False, preview, preview_every)
        img = _read_image(image, "image")
        bg = _read_image(background, "background")
        m = await selection_mask(img, mask, selection)
        v = _check_variant(variant)
        params = {"prompt": prompt, "method": method, "keep_subject": keep_subject, "masked": not sel.is_empty(m)} | opts.__dict__
        job = jobs.submit(
            "background", params,
            lambda hooks: engine.background(img, bg, m, prompt, method, keep_subject, opts=opts, hooks=hooks, variant=v),
        )
        return await respond(job, wait)

    # ------------------------------------------------------------------ upscale

    @app.get("/api/upscale/models", summary="Super-resolution models for /api/upscale")
    async def upscale_models():
        from .upscale import DEFAULT_UPSCALER, MAX_SCALE, MAX_SIDE, UPSCALERS, is_downloaded

        rows = [m.to_dict() | {"downloaded": await asyncio.to_thread(is_downloaded, m)} for m in UPSCALERS.values()]
        return {"default": DEFAULT_UPSCALER, "max_scale": MAX_SCALE, "max_side": MAX_SIDE, "models": rows}

    @app.post("/api/upscale/download", status_code=202, summary="Download a super-resolution model")
    async def upscale_download(model: str = Form(...), wait: bool = Form(False)):
        from .upscale import download, get_upscaler

        try:
            spec = get_upscaler(model)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return await respond(jobs.submit("download", {"variant": spec.name}, lambda hooks: download(spec, hooks.status) and None), wait)

    @app.post("/api/upscale", status_code=202, summary="Upscale an image with AI, by a factor or to a size")
    async def upscale(
        image: UploadFile = File(..., description="Image to upscale"),
        scale: float | None = Form(None, description="Enlargement factor, e.g. 2, 2.5 or 10 (up to 16)"),
        width: int | None = Form(None, description="Target width in pixels (height follows the aspect ratio unless also given)"),
        height: int | None = Form(None, description="Target height in pixels; with width, the image is fitted inside width x height"),
        model: str | None = Form(None, description="Super-resolution model (see /api/upscale/models; default realesrgan-x4plus)"),
        enhance: bool = Form(False, description="First let Qwen restore detail at `resolution` (generative; slower)"),
        prompt: str = Form("", description="enhance: optional description of the image"),
        tile: int = Form(512, description="Tile size in input pixels (lower it if the GPU runs out of memory)"),
        steps: int | None = f["steps"], seed: int | None = f["seed"], cfg: float = f["cfg"],
        negative_prompt: str | None = f["negative_prompt"], resolution: int = f["resolution"],
        variant: str | None = f["variant"], wait: bool = f["wait"],
        preview: str = f["preview"], preview_every: int | None = f["preview_every"],
    ):
        from .upscale import get_upscaler, target_size

        opts = _options(None, None, steps, seed, cfg, negative_prompt, 1, resolution, False, preview if enhance else "off", preview_every)
        img = _read_image(image, "image")
        try:
            get_upscaler(model)
            target = target_size(img.size, scale, width, height)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        if not 64 <= tile <= 2048:
            raise HTTPException(400, "tile must be between 64 and 2048")
        v = _check_variant(variant)
        params = {
            "scale": scale, "width": width, "height": height, "size": list(target), "model": model,
            "enhance": enhance, "prompt": prompt, "tile": tile,
        } | ({k: val for k, val in opts.__dict__.items() if k not in ("width", "height")} if enhance else {})
        job = jobs.submit(
            "upscale", params,
            lambda hooks: engine.upscale(img, scale, width, height, model, enhance, prompt, tile, opts, hooks, v),
        )
        return await respond(job, wait)

    # ------------------------------------------------------------------ jobs

    @app.get("/api/jobs")
    def list_jobs(limit: int = 50):
        return [j.to_dict() | {"queue_position": jobs.queue_position(j)} for j in jobs.list()[:limit]]

    def _job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        return job

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str, wait: bool = False):
        return await respond(_job(job_id), wait)

    @app.delete("/api/jobs/{job_id}")
    def cancel_job(job_id: str):
        _job(job_id)
        return jobs.cancel(job_id).to_dict()

    @app.get("/api/jobs/{job_id}/preview", summary="Latest live preview of a running job (PNG)")
    def job_preview(job_id: str):
        job = _job(job_id)
        if not job.preview_png:
            raise HTTPException(404, "No preview yet")
        return Response(
            job.preview_png,
            media_type="image/png",
            headers={"Cache-Control": "no-store", "X-Preview-Step": str(job.preview_step)},
        )

    @app.get("/api/jobs/{job_id}/extras/{name}", summary="Additional job output, e.g. the control map used")
    def job_extra(job_id: str, name: str):
        job = _job(job_id)
        if name not in job.extras:
            raise HTTPException(404, "No such output")
        return FileResponse(job.extras[name], media_type="image/png")

    @app.get("/api/jobs/{job_id}/images/{index}")
    def job_image(job_id: str, index: int, download: bool = False):
        job = _job(job_id)
        if not 0 <= index < len(job.images):
            raise HTTPException(404, "No such image")
        name = f"{job.kind}-{job.id}-{index}.png"
        return FileResponse(job.images[index], media_type="image/png", filename=name if download else None)

    # ------------------------------------------------------------------ web UI

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
