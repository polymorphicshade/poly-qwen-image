"""Command-line interface.

    qwen-image generate -p "a red fox in snow" -o fox.png
    qwen-image edit -i fox.png -p "make it night" --box 100,100,400,400
    qwen-image pose -i hero.png --pose stick_figure.png --pose-type skeleton
    qwen-image face -i scene.png --face hero_portrait.png [--box 300,80,420,220]
    qwen-image combine -i cat.png -i hat.png -p "the cat wearing the hat"
    qwen-image isolate -i cartoon.png --polygon "10,10 200,15 190,300 20,280" -d "the knight"
    qwen-image background -i portrait.png --background beach.png
    qwen-image models list | download VARIANT
    qwen-image serve --port 7860

Operations run in-process by default (the model loads once per invocation). Pass
``--api http://host:port`` to run them on a server instead, which keeps the model warm.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .variants import DEFAULT_VARIANT, list_variant_names


def _floats(text: str, n: int | None = None) -> list[float]:
    vals = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    if n is not None and len(vals) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers, got '{text}'")
    return vals


def _polygon(text: str) -> list[list[float]]:
    """'x,y x,y x,y' -> [[x, y], ...]"""
    pts = [_floats(p, 2) for p in text.split()]
    if len(pts) < 3:
        raise argparse.ArgumentTypeError("a polygon needs at least 3 points: 'x,y x,y x,y'")
    return pts


def _box(text: str) -> list[float]:
    return _floats(text, 4)


def _add_common(p: argparse.ArgumentParser, size: bool = True):
    g = p.add_argument_group("sampling")
    g.add_argument("--variant", "-m", help=f"model variant (default: {DEFAULT_VARIANT}); see 'models list'")
    if size:
        g.add_argument("--width", type=int)
        g.add_argument("--height", type=int)
    g.add_argument("--resolution", type=int, default=1024, help="target side length when width/height are unset")
    g.add_argument("--steps", type=int, help="denoising steps (default 40; Turbo uses its fixed 8)")
    g.add_argument("--seed", type=int, help="random seed (default: random)")
    g.add_argument("--cfg", type=float, default=1.0, help="true CFG scale; >1 enables guidance")
    g.add_argument("--negative", dest="negative_prompt", help="negative prompt (used when --cfg > 1)")
    g.add_argument("--num-images", "-n", type=int, default=1)
    if size:
        g.add_argument("--transparent", "-t", action="store_true", help="generate a transparent RGBA image")
    g.add_argument("--offload", choices=["auto", "none", "model", "sequential"], default="auto", help="CPU offload strategy")
    g = p.add_argument_group("live previews")
    g.add_argument("--preview", choices=["off", "fast", "full"], help="preview mode (default: fast when --preview-dir is set, else off)")
    g.add_argument("--preview-every", type=int, help="make a preview every N steps (default 1 for fast, 5 for full)")
    g.add_argument("--preview-dir", help="save each step's preview here as step_NN.png")
    p.add_argument("--output", "-o", default=None, help="output file; with several images, _1, _2 ... are appended")
    p.add_argument("--api", help="run on a server, e.g. http://127.0.0.1:7860")
    p.add_argument("--json", action="store_true", help="print a JSON summary to stdout")


def _add_selection(p: argparse.ArgumentParser):
    g = p.add_argument_group("selection (any combination; they are unioned)")
    g.add_argument("--mask", help="mask image: white/opaque = selected")
    g.add_argument("--polygon", action="append", type=_polygon, default=[], help="lasso polygon 'x,y x,y x,y ...' (repeatable)")
    g.add_argument("--box", action="append", type=_box, default=[], help="rectangle 'x0,y0,x1,y1' (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qwen-image", description="Qwen-Image-2.1 generation, editing, composition and subject isolation.")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="text-to-image, optionally with reference images")
    p.add_argument("--prompt", "-p", required=True)
    p.add_argument("--reference", "-r", action="append", default=[], help="reference image (repeatable, up to 10)")
    _add_common(p)

    p = sub.add_parser("edit", help="edit an image, optionally only inside a selection")
    p.add_argument("--image", "-i", required=True)
    p.add_argument("--prompt", "-p", required=True)
    _add_selection(p)
    p.add_argument("--mask-mode", choices=["annotate", "reference"], default="annotate")
    p.add_argument("--no-keep-outside", dest="keep_outside", action="store_false", help="let the model change pixels outside the selection")
    p.add_argument("--feather", type=float, default=8.0)
    _add_common(p)

    p = sub.add_parser("pose", help="re-pose an image's subject to match a pose reference")
    p.add_argument("--image", "-i", required=True, help="image whose subject should be re-posed")
    p.add_argument("--pose", required=True, help="pose reference: a character/photo in the pose, or a stick figure / sketch")
    p.add_argument("--prompt", "-p", default="", help="optional extra instructions")
    p.add_argument("--pose-type", choices=["auto", "character", "skeleton", "sketch"], default="auto")
    p.add_argument("--no-keep-background", dest="keep_background", action="store_false", help="allow the background to change")
    _add_common(p)

    p = sub.add_parser("face", help="give an image's subject the face from a reference image")
    p.add_argument("--image", "-i", required=True, help="image whose subject's face should change")
    p.add_argument("--face", required=True, help="reference image showing the face to use")
    p.add_argument("--prompt", "-p", default="", help="optional extra instructions")
    _add_selection(p)
    p.add_argument("--transfer-hair", dest="keep_hair", action="store_false", help="take the hairstyle from the reference too")
    p.add_argument("--transfer-expression", dest="keep_expression", action="store_false", help="take the expression from the reference too")
    p.add_argument("--no-keep-outside", dest="keep_outside", action="store_false", help="let the model change pixels outside the selection")
    p.add_argument("--feather", type=float, default=6.0)
    _add_common(p)

    p = sub.add_parser("combine", help="combine subjects from 2-10 images into a new image")
    p.add_argument("--image", "-i", action="append", required=True, help="subject image (repeat 2-10 times, in prompt order)")
    p.add_argument("--prompt", "-p", required=True)
    _add_common(p)

    p = sub.add_parser("isolate", help="cut a selected subject out onto a transparent background")
    p.add_argument("--image", "-i", required=True)
    _add_selection(p)
    p.add_argument("--description", "-d", default="", help="what the selection is, e.g. 'the sword the knight holds'")
    p.add_argument("--method", choices=["model", "mask"], default="model", help="'model' = AI matte, 'mask' = selection as-is")
    p.add_argument("--no-crop", dest="crop", action="store_false", help="send the whole image instead of a crop around the selection")
    p.add_argument("--keep-canvas", action="store_true", help="output full-size canvas with the subject in place")
    p.add_argument("--no-clip", dest="clip", action="store_false", help="keep model output outside the selection")
    p.add_argument("--no-restore-pixels", dest="restore_pixels", action="store_false", help="use the model's colours, not the original's")
    _add_common(p, size=False)

    p = sub.add_parser("background", help="replace an image's background with the one from another image")
    p.add_argument("--image", "-i", required=True, help="image whose subject should be kept")
    p.add_argument("--background", "-b", required=True, help="image whose background should be used")
    _add_selection(p)
    p.add_argument("--prompt", "-p", default="", help="model: extra instructions; composite: what the subject is")
    p.add_argument("--method", choices=["model", "composite"], default="model",
                   help="'model' repaints the scene behind the subject; 'composite' cuts it out and pastes it over")
    p.add_argument("--keep-subject", action="store_true", help="model + selection: paste the original pixels inside the selection back")
    _add_common(p)

    p = sub.add_parser("models", help="list or download model variants")
    msub = p.add_subparsers(dest="models_command", required=True)
    m = msub.add_parser("list")
    m.add_argument("--json", action="store_true")
    m = msub.add_parser("download")
    m.add_argument("variant", choices=list_variant_names())

    p = sub.add_parser("serve", help="start the web UI and API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--variant", "-m", help="default variant")
    p.add_argument("--preload", action="store_true", help="load the default variant at startup")
    p.add_argument("--offload", choices=["auto", "none", "model", "sequential"], default="auto")
    p.add_argument("--outputs", default="outputs", help="directory for generated images")
    return ap


# ---------------------------------------------------------------------- helpers


def _open(path: str):
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path))
    return img if img.mode in ("RGB", "RGBA") else img.convert("RGBA" if "A" in img.getbands() else "RGB")


def _output_paths(args, count: int) -> list[Path]:
    if args.output:
        base = Path(args.output)
    else:
        base = Path(f"{args.command}-{time.strftime('%Y%m%d-%H%M%S')}.png")
    if base.suffix == "":
        base = base.with_suffix(".png")
    if count == 1:
        return [base]
    return [base.with_name(f"{base.stem}_{i + 1}{base.suffix}") for i in range(count)]


def _options(args):
    from .engine import Options

    return Options.from_dict(vars(args))


def _resolve_preview(args) -> None:
    if getattr(args, "preview", None) is None:
        args.preview = "fast" if getattr(args, "preview_dir", None) else "off"


def _save_preview(args, image, step) -> None:
    if args.preview_dir:
        folder = Path(args.preview_dir)
        folder.mkdir(parents=True, exist_ok=True)
        image.save(folder / f"step_{step:02d}.png")


def _progress_hooks(args=None):
    from .engine import Hooks

    state = {"mid_bar": False}

    def progress(step, total):
        width = 30
        done = int(width * step / max(1, total))
        sys.stderr.write(f"\r[{'#' * done}{'.' * (width - done)}] {step}/{total}")
        sys.stderr.flush()
        state["mid_bar"] = step < total
        if step >= total:
            sys.stderr.write("\n")

    def status(msg):
        if state["mid_bar"]:
            sys.stderr.write("\n")
            state["mid_bar"] = False
        print(msg, file=sys.stderr)

    hooks = Hooks(progress=progress, status=status)
    if args is not None and getattr(args, "preview_dir", None):
        hooks.preview = lambda image, step: _save_preview(args, image, step)
    return hooks


def _local_mask(args, size):
    from . import selection as sel

    mask = _open(args.mask) if args.mask else None
    return sel.build_mask(size, mask, args.polygon, args.box)


# ---------------------------------------------------------------------- local execution


def run_local(args) -> tuple[list, dict]:
    from .engine import Engine

    engine = Engine(args.variant, offload=args.offload)
    opts, hooks = _options(args), _progress_hooks(args)
    if args.command == "generate":
        r = engine.generate(args.prompt, [_open(p) for p in args.reference], opts, hooks)
    elif args.command == "edit":
        image = _open(args.image)
        r = engine.edit(image, args.prompt, _local_mask(args, image.size), args.mask_mode, args.keep_outside, args.feather, opts, hooks)
    elif args.command == "pose":
        r = engine.pose(_open(args.image), _open(args.pose), args.prompt, args.pose_type, args.keep_background, opts, hooks)
    elif args.command == "face":
        image = _open(args.image)
        r = engine.face(
            image, _open(args.face), _local_mask(args, image.size), args.prompt, args.keep_hair,
            args.keep_expression, args.keep_outside, args.feather, opts, hooks,
        )
    elif args.command == "combine":
        r = engine.combine([_open(p) for p in args.image], args.prompt, opts, hooks)
    elif args.command == "isolate":
        image = _open(args.image)
        mask = _local_mask(args, image.size)
        if mask is None:
            raise SystemExit("isolate needs a selection: --mask, --polygon or --box")
        r = engine.isolate(
            image, mask, args.description, args.method, args.crop, args.keep_canvas, args.clip, args.restore_pixels, opts, hooks
        )
    elif args.command == "background":
        image = _open(args.image)
        r = engine.background(
            image, _open(args.background), _local_mask(args, image.size), args.prompt, args.method, args.keep_subject,
            opts=opts, hooks=hooks,
        )
    else:
        raise AssertionError(args.command)
    return r.images, {"seed": r.seed, "prompt": r.prompt, "variant": r.variant, "info": r.info}


# ---------------------------------------------------------------------- remote execution


def run_remote(args) -> tuple[list, dict]:
    import io

    import httpx
    from PIL import Image

    base = args.api.rstrip("/")
    data: dict = {"wait": "false"}
    keys = ("variant", "width", "height", "resolution", "steps", "seed", "cfg", "negative_prompt", "num_images", "transparent",
            "preview", "preview_every")
    for key in keys:
        val = getattr(args, key, None)
        if val is not None and val is not False:
            data[key] = str(val).lower() if isinstance(val, bool) else str(val)
    files: list = []

    def attach(field, path):
        files.append((field, (Path(path).name, open(path, "rb"), "application/octet-stream")))

    if args.command == "generate":
        data["prompt"] = args.prompt
        for p in args.reference:
            attach("references", p)
    elif args.command == "combine":
        data["prompt"] = args.prompt
        for p in args.image:
            attach("images", p)
    elif args.command == "pose":
        attach("image", args.image)
        attach("pose", args.pose)
        data |= {"prompt": args.prompt, "pose_type": args.pose_type, "keep_background": str(args.keep_background).lower()}
    elif args.command in ("edit", "isolate", "face", "background"):
        attach("image", args.image)
        if args.mask:
            attach("mask", args.mask)
        if args.polygon or args.box:
            data["selection"] = json.dumps({"polygons": args.polygon, "boxes": args.box})
        if args.command == "background":
            attach("background", args.background)
            data |= {"prompt": args.prompt, "method": args.method, "keep_subject": str(args.keep_subject).lower()}
        elif args.command == "face":
            attach("face", args.face)
            data |= {
                "prompt": args.prompt, "keep_hair": str(args.keep_hair).lower(),
                "keep_expression": str(args.keep_expression).lower(),
                "keep_outside": str(args.keep_outside).lower(), "feather": str(args.feather),
            }
        elif args.command == "edit":
            data |= {"prompt": args.prompt, "mask_mode": args.mask_mode, "keep_outside": str(args.keep_outside).lower(), "feather": str(args.feather)}
        else:
            data |= {
                "description": args.description, "method": args.method, "crop": str(args.crop).lower(),
                "keep_canvas": str(args.keep_canvas).lower(), "clip": str(args.clip).lower(),
                "restore_pixels": str(args.restore_pixels).lower(),
            }

    hooks = _progress_hooks(args)
    with httpx.Client(timeout=httpx.Timeout(60.0, read=None)) as client:
        resp = client.post(f"{base}/api/{args.command}", data=data, files=files or None)
        if resp.status_code >= 400:
            raise SystemExit(f"Server error {resp.status_code}: {resp.text}")
        job = resp.json()
        last_msg = None
        last_preview = 0
        while job["status"] not in ("done", "error", "cancelled"):
            time.sleep(0.5)
            job = client.get(f"{base}/api/jobs/{job['id']}").json()
            if job["message"] and job["message"] != last_msg:
                hooks.status(job["message"])
                last_msg = job["message"]
            if job["progress"]["total"]:
                hooks.progress(job["progress"]["step"], job["progress"]["total"])
            if args.preview_dir and job.get("preview") and job["preview_step"] != last_preview:
                last_preview = job["preview_step"]
                hooks.preview(Image.open(io.BytesIO(client.get(base + job["preview"]).content)), last_preview)
        if job["status"] != "done":
            raise SystemExit(f"Job {job['status']}: {job.get('error') or ''}")
        images = [Image.open(io.BytesIO(client.get(base + url).content)) for url in job["images"]]
    return images, {"seed": job["seed"], "prompt": job["prompt"], "variant": job["variant"], "info": job["info"], "job": job["id"]}


# ---------------------------------------------------------------------- commands


def cmd_models(args) -> int:
    from .models import download, is_downloaded
    from .variants import get_variant, list_variants

    if args.models_command == "list":
        rows = [v.to_dict() | {"downloaded": is_downloaded(v)} for v in list_variants()]
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        print(f"{'variant':<14} {'dl':<3} {'~GB':>5}  description")
        for r in rows:
            mark = "yes" if r["downloaded"] else "-"
            default = " (default)" if r["name"] == DEFAULT_VARIANT else ""
            print(f"{r['name']:<14} {mark:<3} {r['approx_download_gb']:>5.1f}  {r['description']}{default}")
        return 0
    download(get_variant(args.variant), lambda msg: print(msg, file=sys.stderr))
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .api import create_app
    from .engine import Engine

    engine = Engine(args.variant or DEFAULT_VARIANT, offload=args.offload)
    if args.preload:
        engine.load(hooks=_progress_hooks())
    app = create_app(engine, args.outputs)
    print(f"Web UI: http://{args.host}:{args.port}/   API docs: http://{args.host}:{args.port}/docs", file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    if args.command == "models":
        return cmd_models(args)
    if args.command == "serve":
        return cmd_serve(args)
    if args.command == "combine" and not 2 <= len(args.image) <= 10:
        raise SystemExit("combine needs between 2 and 10 --image arguments")

    _resolve_preview(args)
    images, meta = run_remote(args) if args.api else run_local(args)
    paths = _output_paths(args, len(images))
    for img, path in zip(images, paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
    meta["outputs"] = [str(p) for p in paths]
    if args.json:
        print(json.dumps(meta, indent=2, default=str))
    else:
        for p in paths:
            print(p)
        print(f"seed: {meta['seed']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
