"""End-to-end tests over the mock variant: engine, HTTP API and CLI, with no model download."""

import io
import json
import os

os.environ["QWEN_IMAGE_ENABLE_MOCK"] = "1"
os.environ["QWEN_IMAGE_MOCK_DELAY"] = "0"

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from qwen_image import selection as sel
from qwen_image.api import create_app
from qwen_image.cli import main as cli_main
from qwen_image.engine import Engine, Options


def make_image(size=(640, 480), color=(40, 120, 200)):
    img = Image.new("RGB", size, color)
    ImageDraw.Draw(img).ellipse([200, 150, 400, 350], fill=(250, 200, 30))
    return img


def png(img):
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def engine():
    return Engine("mock")


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(Engine("mock"), tmp_path / "outputs"))


# ---------------------------------------------------------------- selection


def test_build_mask_unions_sources():
    size = (100, 100)
    m = sel.build_mask(size, polygons=[[(0, 0), (10, 0), (10, 10)]], boxes=[[50, 50, 60, 60]])
    assert m.getpixel((55, 55)) == 255 and m.getpixel((8, 2)) == 255 and m.getpixel((30, 30)) == 0
    assert sel.build_mask(size) is None


def test_mask_from_rgba_uses_alpha():
    painted = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    ImageDraw.Draw(painted).rectangle([10, 10, 20, 20], fill=(255, 255, 255, 255))
    m = sel.mask_from_image(painted, (50, 50))
    assert m.getpixel((15, 15)) == 255 and m.getpixel((40, 40)) == 0


def test_annotate_draws_outline_outside_selection():
    img = make_image()
    mask = sel.mask_from_boxes([[100, 100, 200, 200]], img.size)
    out = sel.annotate(img, mask)
    assert out.getpixel((98, 150))[:3] == (255, 0, 0)  # just outside the box
    assert out.getpixel((150, 150))[:3] == img.getpixel((150, 150))  # inside untouched


# ---------------------------------------------------------------- engine


def test_generate(engine):
    r = engine.generate("a fox", opts=Options(width=512, height=384, seed=7, num_images=2))
    assert len(r.images) == 2 and r.images[0].size == (512, 384) and r.seed == 7
    assert r.images[0].mode == "RGB"


def test_generate_transparent(engine):
    r = engine.generate("a sticker", opts=Options(width=256, height=256, transparent=True))
    assert r.images[0].mode == "RGBA" and sel.has_transparency(r.images[0])
    assert r.prompt.startswith("This is an RGBA image with transparency.")


def test_edit_masked_keeps_outside(engine):
    img = make_image()
    mask = sel.mask_from_boxes([[300, 200, 400, 300]], img.size)
    r = engine.edit(img, "make it red", mask, opts=Options(seed=1))
    out = r.images[0]
    assert out.size == img.size
    assert out.getpixel((10, 10)) == img.getpixel((10, 10))
    assert "outlined in red" in r.prompt


def test_combine_requires_two(engine):
    with pytest.raises(ValueError):
        engine.combine([make_image()], "x")
    r = engine.combine([make_image(), make_image(color=(0, 0, 0))], "both together", opts=Options(resolution=512))
    assert r.images[0].size == (512, 512)


@pytest.mark.parametrize("method", ["model", "mask"])
def test_isolate_removes_everything_outside_selection(engine, method):
    img = make_image()
    poly = [(180, 130), (420, 130), (420, 370), (180, 370)]
    mask = sel.mask_from_polygons([poly], img.size)
    r = engine.isolate(img, mask, "the yellow ball", method=method, keep_canvas=True)
    out = r.images[0]
    assert out.mode == "RGBA" and out.size == img.size
    assert out.getpixel((20, 20))[3] == 0  # background removed
    assert out.getpixel((300, 250))[3] > 200  # subject kept
    assert out.getpixel((300, 250))[:3] == img.getpixel((300, 250))  # original colours restored


def test_isolate_trims_when_not_keeping_canvas(engine):
    img = make_image()
    mask = sel.mask_from_boxes([[250, 200, 350, 300]], img.size)
    out = engine.isolate(img, mask, method="mask").images[0]
    assert out.width < 200 and out.height < 200


# ---------------------------------------------------------------- API


def test_api_models(client):
    data = client.get("/api/models").json()
    assert any(v["name"] == "mock" for v in data["variants"])
    assert any(v["name"].startswith("gguf-") for v in data["variants"])


def _done(client, resp):
    assert resp.status_code == 200, resp.text
    job = resp.json()
    assert job["status"] == "done", job
    return job


def test_api_generate_and_fetch_image(client):
    job = _done(client, client.post("/api/generate", data={"prompt": "a fox", "width": "512", "height": "512", "variant": "mock", "wait": "true"}))
    img = Image.open(io.BytesIO(client.get(job["images"][0]).content))
    assert img.size == (512, 512)
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "done"


def test_api_edit_with_selection_json(client):
    data = {"prompt": "add a hat", "selection": json.dumps({"boxes": [[10, 10, 100, 100]]}), "variant": "mock", "wait": "true"}
    job = _done(client, client.post("/api/edit", data=data, files={"image": ("in.png", png(make_image()), "image/png")}))
    assert job["params"]["masked"] is True


def test_api_combine(client):
    files = [("images", ("a.png", png(make_image()), "image/png")), ("images", ("b.png", png(make_image()), "image/png"))]
    _done(client, client.post("/api/combine", data={"prompt": "together", "variant": "mock", "wait": "true"}, files=files))


def test_api_isolate_with_mask_upload(client):
    img = make_image()
    mask = sel.mask_from_boxes([[180, 130, 420, 370]], img.size)
    files = {"image": ("in.png", png(img), "image/png"), "mask": ("m.png", png(mask), "image/png")}
    job = _done(client, client.post("/api/isolate", data={"description": "ball", "variant": "mock", "wait": "true"}, files=files))
    out = Image.open(io.BytesIO(client.get(job["images"][0]).content))
    assert out.mode == "RGBA"


def test_api_isolate_requires_selection(client):
    r = client.post("/api/isolate", data={"variant": "mock"}, files={"image": ("in.png", png(make_image()), "image/png")})
    assert r.status_code == 400


def test_api_rejects_unknown_variant(client):
    r = client.post("/api/generate", data={"prompt": "x", "variant": "nope"})
    assert r.status_code == 400


def test_api_cancel_queued_job(tmp_path):
    os.environ["QWEN_IMAGE_MOCK_DELAY"] = "0.05"
    try:
        c = TestClient(create_app(Engine("mock"), tmp_path))
        first = c.post("/api/generate", data={"prompt": "slow", "steps": "40", "variant": "mock"}).json()
        second = c.post("/api/generate", data={"prompt": "queued", "variant": "mock"}).json()
        cancelled = c.delete(f"/api/jobs/{second['id']}").json()
        assert cancelled["status"] == "cancelled"
        c.delete(f"/api/jobs/{first['id']}")
        assert c.get(f"/api/jobs/{first['id']}?wait=true").json()["status"] == "cancelled"
    finally:
        os.environ["QWEN_IMAGE_MOCK_DELAY"] = "0"


def test_web_ui_served(client):
    assert "Qwen Image Studio" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200


# ---------------------------------------------------------------- CLI


def test_cli_generate_and_isolate(tmp_path):
    out = tmp_path / "gen.png"
    assert cli_main(["generate", "-p", "a fox", "-m", "mock", "--width", "256", "--height", "256", "-o", str(out)]) == 0
    assert Image.open(out).size == (256, 256)

    src = tmp_path / "src.png"
    make_image().save(src)
    cut = tmp_path / "cut.png"
    assert cli_main(["isolate", "-i", str(src), "-m", "mock", "--polygon", "180,130 420,130 420,370 180,370", "-o", str(cut)]) == 0
    assert Image.open(cut).mode == "RGBA"


def test_cli_edit_and_combine(tmp_path):
    src = tmp_path / "src.png"
    make_image().save(src)
    assert cli_main(["edit", "-i", str(src), "-p", "night", "-m", "mock", "--box", "0,0,100,100", "-o", str(tmp_path / "e.png")]) == 0
    assert cli_main(["combine", "-i", str(src), "-i", str(src), "-p", "two", "-m", "mock", "-n", "2", "-o", str(tmp_path / "c.png")]) == 0
    assert (tmp_path / "c_1.png").exists() and (tmp_path / "c_2.png").exists()


# ---------------------------------------------------------------- ControlNet (mock pipeline; no weights)


def test_render_pose_draws_skeleton():
    from qwen_image.annotators import render_pose, template_pose

    img = render_pose(template_pose(384, 512), 384, 512)
    assert img.size == (384, 512) and img.getbbox() is not None
    assert img.getpixel((0, 0)) == (0, 0, 0)


def test_api_controlnet_render_pose(client):
    from qwen_image.annotators import template_pose

    r = client.post("/api/controlnet/pose/render", json={"pose": template_pose(), "width": 300, "height": 400})
    assert r.status_code == 200 and Image.open(io.BytesIO(r.content)).size == (300, 400)
    assert client.post("/api/controlnet/pose/render", json={"pose": {"people": "x"}}).status_code == 400


def test_api_controlnet_with_pose_returns_control_map(client):
    from qwen_image.annotators import template_pose

    data = {"prompt": "a knight", "control_type": "pose", "pose": json.dumps(template_pose(768, 1024)),
            "resolution": "512", "variant": "mock", "wait": "true"}
    job = _done(client, client.post("/api/controlnet", data=data))
    out = Image.open(io.BytesIO(client.get(job["images"][0]).content))
    assert out.width < out.height  # follows the pose canvas aspect (3:4)
    cmap = Image.open(io.BytesIO(client.get(job["extras"]["control"]).content))
    assert cmap.size == out.size


def test_api_controlnet_references_get_tagged(client):
    files = [("control", ("c.png", png(Image.new("RGB", (512, 512))), "image/png")),
             ("references", ("r.png", png(make_image()), "image/png"))]
    job = _done(client, client.post("/api/controlnet", data={"prompt": "dancing", "variant": "mock", "wait": "true"}, files=files))
    assert "<image1>" in job["prompt"] and job["info"]["strength"] == 0.8


def test_api_controlnet_needs_a_signal(client):
    assert client.post("/api/controlnet", data={"prompt": "x", "variant": "mock"}).status_code == 400
    bad = client.post("/api/controlnet", data={"prompt": "x", "variant": "mock", "pose": "{nope"})
    assert bad.status_code == 400


def test_api_controlnet_detect_canny(client):
    job = _done(client, client.post("/api/controlnet/detect", data={"control_type": "canny", "wait": "true"},
                                    files={"image": ("in.png", png(make_image()), "image/png")}))
    edges = Image.open(io.BytesIO(client.get(job["images"][0]).content))
    assert edges.size == (640, 480) and edges.getbbox() is not None


# ---------------------------------------------------------------- auto selection (mock engine: colour stand-in for SAM)


def test_segmenter_colour_selects_clicked_region():
    from qwen_image.segment import Segmenter

    seg = Segmenter("mock")
    mask, info = seg.segment(make_image(), [[300, 250]])
    assert mask.getbbox() == (200, 150, 401, 351)  # the yellow ball
    cut, _ = seg.segment(make_image(), [[300, 250], [20, 20]], [1, 0])
    assert cut.getbbox() == mask.getbbox()
    with pytest.raises(ValueError):
        seg.segment(make_image(), [[1, 1]], [0])
    with pytest.raises(ValueError):
        seg.segment(make_image(), [[1, 1]], [1, 1])


def test_api_segment_and_image_id_reuse(client):
    files = {"image": ("in.png", png(make_image()), "image/png")}
    r = client.post("/api/segment", data={"points": "[[300, 250]]"}, files=files)
    assert r.status_code == 200, r.text
    mask = Image.open(io.BytesIO(r.content))
    assert mask.mode == "L" and mask.getbbox() == (200, 150, 401, 351)
    assert r.headers["X-Bbox"] == "200,150,401,351"
    image_id = r.headers["X-Image-Id"]

    # A second click refers to the image by id; the background point selects the background.
    r = client.post("/api/segment", data={"image_id": image_id, "points": "[[10, 10]]", "output": "alpha"})
    assert r.status_code == 200, r.text
    alpha = Image.open(io.BytesIO(r.content))
    assert alpha.mode == "RGBA" and alpha.getpixel((10, 10))[3] == 255 and alpha.getpixel((300, 250))[3] == 0

    assert client.post("/api/segment", data={"image_id": "nope", "points": "[[1, 1]]"}).status_code == 404
    assert client.post("/api/segment", data={"points": "[[1, 1]]"}).status_code == 400
    bad = client.post("/api/segment", data={"image_id": image_id, "points": "[[1, 1]]", "labels": "[0]"})
    assert bad.status_code == 400


def test_api_isolate_with_auto_selected_point(client):
    files = {"image": ("in.png", png(make_image()), "image/png")}
    data = {"selection": json.dumps({"points": [[300, 250]]}), "method": "mask", "keep_canvas": "true", "variant": "mock", "wait": "true"}
    job = _done(client, client.post("/api/isolate", data=data, files=files))
    out = Image.open(io.BytesIO(client.get(job["images"][0]).content))
    assert out.getpixel((300, 250))[3] == 255 and out.getpixel((20, 20))[3] == 0


def test_api_models_lists_segmenter(client):
    assert client.get("/api/models").json()["segment"]["repo"] == "mock"


def test_cli_segment_and_point_selection(tmp_path):
    src = tmp_path / "src.png"
    make_image().save(src)
    mask = tmp_path / "mask.png"
    assert cli_main(["segment", "-i", str(src), "--point", "300,250", "--method", "color", "-o", str(mask)]) == 0
    assert Image.open(mask).getbbox() == (200, 150, 401, 351)
    cut = tmp_path / "cut.png"
    assert cli_main(["isolate", "-i", str(src), "-m", "mock", "--point", "300,250", "--method", "mask", "-o", str(cut)]) == 0
    assert Image.open(cut).size[0] < 220
