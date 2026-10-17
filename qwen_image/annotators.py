"""Control-map preprocessors ("annotators") for the ControlNet-Union.

Every map follows the convention the ControlNet was trained on: white/coloured features on black
(depth: near is bright). Pose maps are OpenPose/DWPose-style skeletons rendered from keypoints, so
detected poses can be edited and re-rendered.

Pose keypoint JSON (all coordinates normalised to 0..1, ``null`` = not visible)::

    {"width": W, "height": H, "people": [
        {"body": [[x, y] | null] * 18, "left_hand": [[x, y] | null] * 21 | null,
         "right_hand": ... | null, "face": [[x, y] | null] * 70 | null}
    ]}

Body order is OpenPose COCO-18: nose, neck, r-shoulder, r-elbow, r-wrist, l-shoulder, l-elbow,
l-wrist, r-hip, r-knee, r-ankle, l-hip, l-knee, l-ankle, r-eye, l-eye, r-ear, l-ear.
"""

from __future__ import annotations

import colorsys
import logging
import math
import threading

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)

ANNOTATOR_REPO = "lllyasviel/Annotators"
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"

CONTROL_TYPES = {
    "pose": "Pose (skeleton)",
    "canny": "Canny edges",
    "depth": "Depth",
    "lineart": "Line art",
    "softedge": "Soft edges (HED)",
    "mlsd": "Straight lines (MLSD)",
    "scribble": "Scribble",
    "grayscale": "Grayscale",
}

# OpenPose COCO-18 limbs (1-based keypoint indices) and colours, as in the ControlNet annotators.
LIMBS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13],
         [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
          [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
          [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
              [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

# A neutral standing figure, used as a starting point for drawing a pose by hand.
TEMPLATE_BODY = [
    [0.50, 0.14], [0.50, 0.24], [0.42, 0.24], [0.39, 0.38], [0.37, 0.51], [0.58, 0.24], [0.61, 0.38],
    [0.63, 0.51], [0.45, 0.52], [0.45, 0.71], [0.45, 0.90], [0.55, 0.52], [0.55, 0.71], [0.55, 0.90],
    [0.48, 0.12], [0.52, 0.12], [0.46, 0.13], [0.54, 0.13],
]


def template_pose(width: int = 768, height: int = 1024) -> dict:
    return {"width": width, "height": height, "people": [{"body": [list(p) for p in TEMPLATE_BODY]}]}


# ---------------------------------------------------------------------- pose rendering


def _pt(p, w, h):
    if p is None or len(p) < 2 or p[0] is None or p[0] < 0 or p[1] < 0:
        return None
    return float(p[0]) * w, float(p[1]) * h


def render_pose(pose: dict, width: int, height: int, hands: bool = True, face: bool = True) -> Image.Image:
    """Draw OpenPose-style skeletons (body limbs as ellipses, joints as dots) on black."""
    import cv2

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    s = max(width, height) / 512.0  # annotators draw at ~512 px; keep the same relative stroke weight
    stick, joint_r = max(2, round(4 * s)), max(2, round(4 * s))
    for person in pose.get("people", []):
        body = person.get("body") or []
        body = (list(body) + [None] * 18)[:18]
        for (a, b), color in zip(LIMBS, COLORS):
            p1, p2 = _pt(body[a - 1], width, height), _pt(body[b - 1], width, height)
            if p1 is None or p2 is None:
                continue
            mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            length = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            angle = math.degrees(math.atan2(p1[1] - p2[1], p1[0] - p2[0]))
            poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length / 2), stick), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, [int(c * 0.6) for c in color])
        for p, color in zip(body, COLORS):
            q = _pt(p, width, height)
            if q is not None:
                cv2.circle(canvas, (int(q[0]), int(q[1])), joint_r, color, thickness=-1)
        if hands:
            for key in ("left_hand", "right_hand"):
                pts = person.get(key)
                if not pts:
                    continue
                pts = [_pt(p, width, height) for p in pts]
                for i, (a, b) in enumerate(HAND_EDGES):
                    if a < len(pts) and b < len(pts) and pts[a] and pts[b]:
                        rgb = colorsys.hsv_to_rgb(i / len(HAND_EDGES), 1.0, 1.0)
                        cv2.line(canvas, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                                 [int(c * 255) for c in rgb], thickness=max(1, round(2 * s)))
                for q in pts:
                    if q:
                        cv2.circle(canvas, (int(q[0]), int(q[1])), max(1, round(3 * s)), (0, 0, 255), thickness=-1)
        if face and person.get("face"):
            for p in person["face"]:
                q = _pt(p, width, height)
                if q:
                    cv2.circle(canvas, (int(q[0]), int(q[1])), max(1, round(2.5 * s)), (255, 255, 255), thickness=-1)
    return Image.fromarray(canvas)


def validate_pose(pose: dict) -> dict:
    if not isinstance(pose, dict) or not isinstance(pose.get("people"), list):
        raise ValueError("pose must be an object with a 'people' list")
    for person in pose["people"]:
        if not isinstance(person, dict) or not isinstance(person.get("body"), list):
            raise ValueError("each person needs a 'body' list of 18 [x, y] points (or null)")
    return pose


# ---------------------------------------------------------------------- detectors


def _np_rgb(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def _short_side(image: Image.Image, target: int) -> Image.Image:
    w, h = image.size
    k = target / min(w, h)
    return image.resize((max(64, round(w * k)), max(64, round(h * k))), Image.Resampling.LANCZOS)


def _kp_list(points, n):
    out = []
    for p in (points or [])[:n]:
        if p is None or getattr(p, "x", -1) < 0 or getattr(p, "y", -1) < 0:
            out.append(None)
        else:
            out.append([round(float(p.x), 5), round(float(p.y), 5)])
    return out or None


class Annotators:
    """Lazily loaded detectors; weights download from the Hub on first use."""

    def __init__(self):
        self._models: dict = {}
        self._lock = threading.Lock()

    @staticmethod
    def _device():
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get(self, name, factory):
        with self._lock:
            if name not in self._models:
                log.info("Loading annotator %s", name)
                self._models[name] = factory()
            return self._models[name]

    def _on_device(self, model):
        """Run a detector on the GPU only for the duration of the call."""
        import contextlib

        import torch

        @contextlib.contextmanager
        def ctx():
            dev = self._device()
            if hasattr(model, "to"):
                model.to(dev)
            try:
                yield model
            finally:
                if hasattr(model, "to") and dev != "cpu":
                    model.to("cpu")
                    torch.cuda.empty_cache()

        return ctx()

    def detect(self, kind: str, image: Image.Image, **opt) -> tuple[Image.Image, dict]:
        """Return (control map at the image's size, extra info - the keypoints for pose)."""
        if kind not in CONTROL_TYPES:
            raise ValueError(f"control type must be one of {', '.join(CONTROL_TYPES)}")
        image = ImageOps.exif_transpose(image).convert("RGB")
        size = image.size
        info: dict = {}
        if kind == "grayscale":
            out = ImageOps.grayscale(image).convert("RGB")
        elif kind == "canny":
            import cv2

            low, high = int(opt.get("low_threshold", 100)), int(opt.get("high_threshold", 200))
            edges = cv2.Canny(cv2.cvtColor(_np_rgb(image), cv2.COLOR_RGB2GRAY), low, high)
            out = Image.fromarray(edges).convert("RGB")
        elif kind == "depth":
            out = self._depth(image)
        elif kind == "pose":
            pose = self.detect_pose(image, hands=opt.get("hands", True), face=opt.get("face", True))
            info["pose"] = pose
            out = render_pose(pose, *size)
        else:
            out = self._aux(kind, image, opt)
        if out.size != size:
            out = out.resize(size, Image.Resampling.BILINEAR)
        return out.convert("RGB"), info

    def detect_pose(self, image: Image.Image, hands: bool = True, face: bool = True) -> dict:
        from controlnet_aux import OpenposeDetector

        model = self._get("openpose", lambda: OpenposeDetector.from_pretrained(ANNOTATOR_REPO))
        small = _short_side(image.convert("RGB"), 512)
        with self._on_device(model) as m:
            results = m.detect_poses(_np_rgb(small), include_hand=hands, include_face=face)
        people = []
        for r in results:
            person = {"body": _kp_list(r.body.keypoints, 18) or [None] * 18}
            if hands:
                person["left_hand"] = _kp_list(r.left_hand, 21)
                person["right_hand"] = _kp_list(r.right_hand, 21)
            if face:
                person["face"] = _kp_list(r.face, 70)
            people.append(person)
        return {"width": image.width, "height": image.height, "people": people}

    def _depth(self, image: Image.Image) -> Image.Image:
        from transformers import pipeline

        pipe = self._get("depth", lambda: pipeline("depth-estimation", model=DEPTH_REPO, device="cpu"))
        with self._on_device(pipe.model):
            pipe.device = pipe.model.device
            depth = pipe(_short_side(image, 518))["depth"]
        return ImageOps.autocontrast(depth.convert("L")).convert("RGB")

    def _aux(self, kind: str, image: Image.Image, opt: dict) -> Image.Image:
        import controlnet_aux as aux

        res = int(opt.get("detect_resolution", 768))
        if kind in ("softedge", "scribble"):
            model = self._get("hed", lambda: aux.HEDdetector.from_pretrained(ANNOTATOR_REPO))
            kwargs = {"scribble": kind == "scribble", "safe": kind == "softedge"}
        elif kind == "lineart":
            model = self._get("lineart", lambda: aux.LineartDetector.from_pretrained(ANNOTATOR_REPO))
            kwargs = {"coarse": bool(opt.get("coarse", False))}
        elif kind == "mlsd":
            model = self._get("mlsd", lambda: aux.MLSDdetector.from_pretrained(ANNOTATOR_REPO))
            kwargs = {"thr_v": float(opt.get("thr_v", 0.1)), "thr_d": float(opt.get("thr_d", 0.1))}
        else:
            raise ValueError(kind)
        with self._on_device(model) as m:
            return m(image, detect_resolution=res, image_resolution=res, output_type="pil", **kwargs)


def prepare_map(image: Image.Image, invert: bool = False) -> Image.Image:
    """Normalise a user-supplied control map; ``invert`` turns black-on-white drawings into white-on-black."""
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        bg.alpha_composite(rgba)
        image = bg
    image = image.convert("RGB")
    return ImageOps.invert(image) if invert else image
