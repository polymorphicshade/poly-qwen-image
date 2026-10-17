"use strict";

// ControlNet tab: the control-map stage and the interactive pose (skeleton) editor.
// Pose keypoints are normalised 0..1 in OpenPose COCO-18 order (see qwen_image/annotators.py);
// the server renders the exact control map from them, this only draws an on-screen preview.

const POSE_LIMBS = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13],
  [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]];
const POSE_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
  [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
  [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]];
const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
  [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]];
// Left/right body keypoint pairs, for mirroring.
const MIRROR_PAIRS = [[2, 5], [3, 6], [4, 7], [8, 11], [9, 12], [10, 13], [14, 15], [16, 17]];
const TEMPLATE_BODY = [
  [0.50, 0.14], [0.50, 0.24], [0.42, 0.24], [0.39, 0.38], [0.37, 0.51], [0.58, 0.24], [0.61, 0.38],
  [0.63, 0.51], [0.45, 0.52], [0.45, 0.71], [0.45, 0.90], [0.55, 0.52], [0.55, 0.71], [0.55, 0.90],
  [0.48, 0.12], [0.52, 0.12], [0.46, 0.13], [0.54, 0.13],
];
const JOINT_NAMES = ["nose", "neck", "right shoulder", "right elbow", "right wrist", "left shoulder", "left elbow",
  "left wrist", "right hip", "right knee", "right ankle", "left hip", "left knee", "left ankle", "right eye",
  "left eye", "right ear", "left ear"];
const CONTROL_HINTS = {
  pose: "Extract a skeleton from a photo or drawing, or click “Add figure”. Drag joints to adjust; Shift-drag moves the whole figure; double-click hides a joint.",
  canny: "Sharp edges from the source. Good for keeping exact outlines and details.",
  depth: "Near is bright, far is dark. Keeps the 3D layout while changing everything else.",
  lineart: "Clean line drawing of the source. Good for colouring sketches and line art.",
  softedge: "Soft outlines (HED). Looser than Canny, keeps shapes but allows new details.",
  mlsd: "Straight lines only. Good for architecture and interiors.",
  scribble: "Rough strokes. Use a quick sketch (tick “already a control map”, and “Invert” if it is black on white).",
  grayscale: "The source's brightness. Re-colours or re-styles an image while keeping its tones.",
};

function hsv(h) {
  const i = Math.floor(h * 6), f = h * 6 - i, q = 1 - f;
  return [[1, f, 0], [q, 1, 0], [0, 1, f], [0, q, 1], [f, 0, 1], [1, 0, q]][i % 6].map(v => Math.round(v * 255));
}

class ControlStage {
  constructor(ws) {
    this.ws = ws;
    this.canvas = ws.r("cn-canvas");
    this.ctx = this.canvas.getContext("2d");
    this.source = null;   // { blob, bitmap }
    this.map = null;      // { blob, bitmap, type }
    this.pose = null;     // keypoints JSON
    this.selected = 0;    // index of the figure being edited
    this.drag = null;
    this.bind();
  }

  type() { return this.ws.r("cn-type").value; }
  isMap() { return this.ws.r("cn-is-map").checked; }

  bind() {
    const r = role => this.ws.r(role);
    r("cn-type").addEventListener("change", () => {
      if (this.map && this.map.type !== this.type()) this.map = null;  // a depth map is no use as a canny map
      this.sync();
    });
    r("cn-is-map").addEventListener("change", () => this.sync());
    r("cn-invert").addEventListener("change", () => this.render());
    r("cn-show-source").addEventListener("change", () => this.render());
    r("cn-extract").onclick = () => this.extract();
    r("cn-clear").onclick = () => { this.map = null; this.pose = null; this.sync(); };
    r("cn-add-figure").onclick = () => this.addFigure();
    r("cn-del-figure").onclick = () => this.deleteFigure();
    r("cn-mirror").onclick = () => this.mirror();
    r("cn-auto-strength").addEventListener("change", () => this.sync());
    r("cn-strength").addEventListener("input", () => this.sync());
    r("cn-inpaint").addEventListener("change", () => { this.ws.updateVisibility(); this.ws.render(); });
    this.bindPointer();
  }

  // --- inputs
  async setSource(blob) {
    this.source = blob ? { blob, bitmap: await createImageBitmap(blob) } : null;
    if (this.map && this.map.fromSource) this.map = null;
    this.sync();
  }

  async setMapBlob(blob, type) {
    this.map = { blob, bitmap: await createImageBitmap(blob), type };
    this.sync();
  }

  setPose(pose) {
    this.pose = pose && pose.people ? pose : null;
    this.selected = 0;
    this.sync();
  }

  // Width/height of the control signal (drives the canvas aspect and the default output aspect).
  dims() {
    if (this.type() === "pose" && this.pose) return [this.pose.width || 768, this.pose.height || 1024];
    const bmp = (this.map && this.map.bitmap) || (this.source && this.source.bitmap);
    return bmp ? [bmp.width, bmp.height] : [768, 1024];
  }

  sync() {
    const ws = this.ws, pose = this.type() === "pose";
    ws.r("cn-pose-tools").hidden = !pose;
    ws.r("cn-hands-row").hidden = !pose;
    ws.r("cn-invert-row").hidden = !this.isMap();
    ws.r("cn-extract").disabled = !this.source || this.isMap();
    const auto = ws.r("cn-auto-strength").checked;
    ws.r("cn-strength").disabled = auto;
    ws.r("cn-strength-val").textContent = auto ? (ws.multi.length ? "auto (0.80)" : "auto (1.00)") : Number(ws.r("cn-strength").value).toFixed(2);
    ws.r("cn-hint").textContent = CONTROL_HINTS[this.type()] || "";
    const empty = !this.hasSignal();
    ws.r("cn-empty").hidden = !empty;
    ws.r("cn-empty").textContent = this.source && !this.isMap()
      ? `Click “Extract control map” to preview the ${this.type()} map (or just run — it is extracted automatically).`
      : pose ? "Add a control source and extract its pose, or click “Add figure” to draw one." : "Add a control source on the left.";
    this.render();
  }

  hasSignal() {
    if (this.type() === "pose" && this.pose && this.pose.people.length) return true;
    return !!(this.map || (this.source && this.isMap()));
  }

  // --- extraction
  async extract() {
    if (!this.source) return;
    const ws = this.ws, type = this.type(), btn = ws.r("cn-extract");
    const form = new FormData();
    form.append("image", this.source.blob, "source.png");
    form.append("control_type", type);
    const hands = ws.r("cn-hands").checked;
    if (type === "pose") form.append("options", JSON.stringify({ hands, face: hands }));
    btn.disabled = true;
    ws.setStatus(`Extracting ${type} map…`);
    try {
      const job = await api.post("/api/controlnet/detect", form);
      const done = await api.waitJob(job.id, j => ws.setStatus(j.status === "queued" ? `Queued (position ${j.queue_position})…` : (j.message || "Extracting…")));
      if (done.status !== "done") throw new Error(done.error || done.status);
      const blob = await (await fetch(done.images[0])).blob();
      this.map = { blob, bitmap: await createImageBitmap(blob), type, fromSource: true };
      if (type === "pose") {
        this.setPose(done.info.pose);
        ws.setStatus(this.pose.people.length ? `Found ${this.pose.people.length} figure(s). Drag joints to adjust.` : "No figure found — try another image, or add one by hand.", !this.pose.people.length);
      } else {
        ws.setStatus(`${type} map ready.`);
        this.sync();
      }
    } catch (e) {
      ws.setStatus(`Extraction failed: ${e.message}`, true);
    } finally {
      btn.disabled = !this.source || this.isMap();
    }
  }

  // --- form
  // Adds the control signal to a generation request; returns false when there is none.
  appendTo(form) {
    const type = this.type();
    if (type === "pose" && this.pose && this.pose.people.length) {
      const hands = this.ws.r("cn-hands").checked;
      const people = this.pose.people.map(p => hands ? p : { body: p.body });
      form.append("pose", JSON.stringify({ width: this.pose.width, height: this.pose.height, people }));
      return true;
    }
    if (this.source && this.isMap()) {
      form.append("control", this.source.blob, "control.png");
      form.append("invert", this.ws.r("cn-invert").checked);
      return true;
    }
    if (this.map && this.map.type === type) {
      form.append("control", this.map.blob, "control.png");
      return true;
    }
    if (this.source) {
      form.append("source", this.source.blob, "source.png");
      if (type === "pose") {
        const hands = this.ws.r("cn-hands").checked;
        form.append("detect_options", JSON.stringify({ hands, face: hands }));
      }
      return true;
    }
    return false;
  }

  // --- figures
  ensurePose() {
    if (!this.pose) {
      const [w, h] = this.source ? [this.source.bitmap.width, this.source.bitmap.height] : [768, 1024];
      this.pose = { width: w, height: h, people: [] };
    }
  }

  addFigure() {
    this.ensurePose();
    const n = this.pose.people.length, aspect = this.pose.width / this.pose.height;
    // The template is drawn for a 3:4 canvas; keep its proportions on other aspects.
    const sx = Math.min(1, 0.75 / aspect), dx = (n % 3) * 0.12 - (n ? 0.12 : 0);
    const body = TEMPLATE_BODY.map(([x, y]) => [0.5 + (x - 0.5) * sx + dx, y]);
    this.pose.people.push({ body });
    this.selected = this.pose.people.length - 1;
    this.sync();
  }

  deleteFigure() {
    if (!this.pose || !this.pose.people.length) return;
    this.pose.people.splice(this.selected, 1);
    this.selected = Math.max(0, this.selected - 1);
    this.sync();
  }

  mirror() {
    const p = this.pose && this.pose.people[this.selected];
    if (!p) return;
    const flip = pts => pts && pts.map(q => q ? [1 - q[0], q[1]] : q);
    p.body = flip(p.body);
    for (const [a, b] of MIRROR_PAIRS) [p.body[a], p.body[b]] = [p.body[b], p.body[a]];
    [p.left_hand, p.right_hand] = [flip(p.right_hand), flip(p.left_hand)];
    p.face = flip(p.face);
    this.render();
  }

  // --- drawing
  layout() {
    const [w, h] = this.dims();
    const k = 1024 / Math.max(w, h);
    const cw = Math.round(w * k), ch = Math.round(h * k);
    if (this.canvas.width !== cw || this.canvas.height !== ch) { this.canvas.width = cw; this.canvas.height = ch; }
    return [cw, ch];
  }

  render() {
    if (this.ws.mode !== "controlnet") return;
    const [W, H] = this.layout();
    const ctx = this.ctx;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    const showSource = this.ws.r("cn-show-source").checked;
    const pose = this.type() === "pose";
    if (pose && this.pose) {
      if (showSource && this.source) { ctx.globalAlpha = 0.35; ctx.drawImage(this.source.bitmap, 0, 0, W, H); ctx.globalAlpha = 1; }
      this.drawPose(ctx, W, H);
      return;
    }
    const mapBmp = this.map && this.map.type === this.type() ? this.map.bitmap : (this.source && this.isMap() ? this.source.bitmap : null);
    if (mapBmp) {
      if (this.isMap() && this.ws.r("cn-invert").checked) ctx.filter = "invert(1)";
      ctx.drawImage(mapBmp, 0, 0, W, H);
      ctx.filter = "none";
      if (showSource && this.source && !this.isMap()) { ctx.globalAlpha = 0.25; ctx.drawImage(this.source.bitmap, 0, 0, W, H); ctx.globalAlpha = 1; }
    } else if (this.source) {
      ctx.globalAlpha = 0.35;
      ctx.drawImage(this.source.bitmap, 0, 0, W, H);
      ctx.globalAlpha = 1;
    }
  }

  drawPose(ctx, W, H) {
    const s = Math.max(W, H) / 512, stick = Math.max(2, 4 * s);
    const P = q => (q && q[0] >= 0 && q[1] >= 0 ? [q[0] * W, q[1] * H] : null);
    const hands = this.ws.r("cn-hands").checked;
    this.pose.people.forEach((person, pi) => {
      const body = person.body || [];
      ctx.lineCap = "round";
      POSE_LIMBS.forEach(([a, b], i) => {
        const p1 = P(body[a - 1]), p2 = P(body[b - 1]);
        if (!p1 || !p2) return;
        const c = POSE_COLORS[i];
        ctx.strokeStyle = `rgb(${c.map(v => Math.round(v * 0.6)).join(",")})`;
        ctx.lineWidth = stick * 2;
        ctx.beginPath(); ctx.moveTo(...p1); ctx.lineTo(...p2); ctx.stroke();
      });
      body.forEach((q, i) => {
        const p = P(q);
        if (!p) return;
        ctx.fillStyle = `rgb(${POSE_COLORS[i].join(",")})`;
        ctx.beginPath(); ctx.arc(p[0], p[1], stick, 0, Math.PI * 2); ctx.fill();
        if (pi === this.selected) {
          ctx.strokeStyle = "rgba(255,255,255,0.85)";
          ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(p[0], p[1], stick + 4, 0, Math.PI * 2); ctx.stroke();
        }
      });
      if (!hands) return;
      for (const key of ["left_hand", "right_hand"]) {
        const pts = (person[key] || []).map(P);
        if (!pts.length) continue;
        HAND_EDGES.forEach(([a, b], i) => {
          if (!pts[a] || !pts[b]) return;
          ctx.strokeStyle = `rgb(${hsv(i / HAND_EDGES.length).join(",")})`;
          ctx.lineWidth = Math.max(1, 2 * s);
          ctx.beginPath(); ctx.moveTo(...pts[a]); ctx.lineTo(...pts[b]); ctx.stroke();
        });
        ctx.fillStyle = "rgb(0,0,255)";
        pts.forEach(p => { if (p) { ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(1, 3 * s), 0, Math.PI * 2); ctx.fill(); } });
      }
      ctx.fillStyle = "#fff";
      (person.face || []).map(P).forEach(p => { if (p) { ctx.beginPath(); ctx.arc(p[0], p[1], Math.max(1, 2.5 * s), 0, Math.PI * 2); ctx.fill(); } });
    });
  }

  // --- pose editing
  toNorm(e) {
    const rect = this.canvas.getBoundingClientRect();
    return [(e.clientX - rect.left) / rect.width, (e.clientY - rect.top) / rect.height];
  }

  hit(e) {
    if (!this.pose) return null;
    const rect = this.canvas.getBoundingClientRect();
    const [x, y] = this.toNorm(e);
    let best = null, bestD = 14;  // CSS pixels
    // Prefer the selected figure when joints overlap.
    const order = this.pose.people.map((_, i) => i).sort((a, b) => (b === this.selected) - (a === this.selected));
    for (const pi of order) {
      (this.pose.people[pi].body || []).forEach((q, j) => {
        if (!q) return;
        const d = Math.hypot((q[0] - x) * rect.width, (q[1] - y) * rect.height);
        if (d < bestD) { bestD = d; best = { pi, j }; }
      });
    }
    return best;
  }

  bindPointer() {
    const c = this.canvas;
    c.addEventListener("pointerdown", e => {
      if (this.type() !== "pose" || e.button !== 0) return;
      const h = this.hit(e);
      if (!h) return;
      c.setPointerCapture(e.pointerId);
      this.selected = h.pi;
      this.drag = { ...h, last: this.toNorm(e), whole: e.shiftKey };
      this.render();
    });
    c.addEventListener("pointermove", e => {
      if (!this.drag) {
        if (this.type() === "pose") {
          const h = this.hit(e);
          c.style.cursor = h ? "grab" : "default";
          c.title = h ? `${JOINT_NAMES[h.j]} — drag to move, Shift-drag moves the figure, double-click hides` : "";
        }
        return;
      }
      const [x, y] = this.toNorm(e);
      const dx = x - this.drag.last[0], dy = y - this.drag.last[1];
      this.drag.last = [x, y];
      const person = this.pose.people[this.drag.pi];
      const shift = pts => pts && pts.forEach(q => { if (q) { q[0] += dx; q[1] += dy; } });
      if (this.drag.whole) {
        shift(person.body); shift(person.left_hand); shift(person.right_hand); shift(person.face);
      } else {
        const q = person.body[this.drag.j];
        q[0] += dx; q[1] += dy;
        // Hands follow their wrists, the face follows the nose.
        if (this.drag.j === 4) shift(person.right_hand);
        if (this.drag.j === 7) shift(person.left_hand);
        if (this.drag.j === 0) shift(person.face);
      }
      this.render();
    });
    const end = () => { this.drag = null; };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("dblclick", e => {
      if (this.type() !== "pose") return;
      const h = this.hit(e);
      if (!h) return;
      this.pose.people[h.pi].body[h.j] = null;
      this.render();
    });
  }
}
