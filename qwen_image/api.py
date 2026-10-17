"""HTTP API (and the static web UI that uses it).

Every operation is a job: ``POST`` returns the job immediately (``202``) and clients poll
``GET /api/jobs/{id}``; pass ``wait=true`` to block until it finishes instead. Images are sent
as multipart file fields so the API is easy to drive with ``curl -F``.
"""

from __future__ import annotations

import asyncio
import io
import logging
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError

from . import __version__
from . import selection as sel
from .engine import Engine, Options
from .jobs import TERMINAL, Job, JobManager
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


def _selection_mask(size, mask: UploadFile | None, selection: str | None) -> Image.Image | None:
    mask_img = _read_image(mask, "mask") if mask is not None and mask.filename else None
    polygons, boxes = [], []
    if selection:
        try:
            parsed = sel.parse_selection_json(selection)
        except ValueError as e:
            raise HTTPException(400, f"Invalid selection: {e}") from e
        polygons, boxes = parsed["polygons"], parsed["boxes"]
    return sel.build_mask(size, mask_img, polygons, boxes)


def _options(
    width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent
) -> Options:
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
        )
    )


def _check_variant(variant: str | None) -> str | None:
    if variant:
        try:
            get_variant(variant)
        except KeyError as e:
            raise HTTPException(400, str(e.args[0])) from e
    return variant or None


def create_app(engine: Engine | None = None, output_dir: str | Path = "outputs") -> FastAPI:
    engine = engine or Engine(DEFAULT_VARIANT)
    jobs = JobManager(engine, output_dir)
    app = FastAPI(title="poly-qwen-image", version=__version__, description="Qwen-Image-2.1 generation, editing, composition and subject isolation.")
    app.state.engine, app.state.jobs = engine, jobs

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
            height=Form(None),
            steps=Form(None, description="Denoising steps (ignored by Turbo variants)"),
            seed=Form(None, description="Random seed; omit or -1 for random"),
            cfg=Form(1.0, description="True CFG scale; >1 enables guidance with negative_prompt"),
            negative_prompt=Form(None),
            num_images=Form(1),
            resolution=Form(1024, description="Target side length when width/height are not given"),
            transparent=Form(False, description="Generate a transparent (RGBA) image"),
            variant=Form(None, description="Model variant to use (default: currently loaded / server default)"),
            wait=Form(False, description="Block until the job finishes"),
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
        return {
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
    ):
        opts = _options(width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent)
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
    ):
        if mask_mode not in ("annotate", "reference"):
            raise HTTPException(400, "mask_mode must be 'annotate' or 'reference'")
        opts = _options(width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent)
        img = _read_image(image, "image")
        m = _selection_mask(img.size, mask, selection)
        v = _check_variant(variant)
        params = {"prompt": prompt, "mask_mode": mask_mode, "masked": not sel.is_empty(m), "keep_outside": keep_outside} | opts.__dict__
        job = jobs.submit(
            "edit", params,
            lambda hooks: engine.edit(img, prompt, m, mask_mode, keep_outside, feather, opts, hooks, v),
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
    ):
        opts = _options(width, height, steps, seed, cfg, negative_prompt, num_images, resolution, transparent)
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
    ):
        if method not in ("model", "mask"):
            raise HTTPException(400, "method must be 'model' or 'mask'")
        opts = _options(None, None, steps, seed, cfg, negative_prompt, num_images, resolution, True)
        img = _read_image(image, "image")
        m = _selection_mask(img.size, mask, selection)
        if sel.is_empty(m):
            raise HTTPException(400, "isolate needs a selection: upload 'mask' or pass 'selection'")
        v = _check_variant(variant)
        params = {"description": description, "method": method, "crop": crop, "keep_canvas": keep_canvas, "clip": clip} | opts.__dict__
        job = jobs.submit(
            "isolate", params,
            lambda hooks: engine.isolate(img, m, description, method, crop, keep_canvas, clip, restore_pixels, opts, hooks, v),
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
