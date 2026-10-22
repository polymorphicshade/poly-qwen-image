"use strict";

// ----------------------------------------------------------------------------- API client

const api = {
  async json(path, opts = {}) {
    const res = await fetch(path, opts);
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON */ }
    if (!res.ok) {
      const detail = body && (body.detail || body.error);
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail) || `HTTP ${res.status}`);
    }
    return body;
  },
  get(path) { return this.json(path); },
  post(path, form) { return this.json(path, { method: "POST", body: form }); },
  del(path) { return this.json(path, { method: "DELETE" }); },
  async waitJob(id, onUpdate, signal) {
    for (;;) {
      const job = await this.get(`/api/jobs/${id}`);
      onUpdate(job);
      if (["done", "error", "cancelled"].includes(job.status)) return job;
      if (signal && signal.aborted) return job;
      await new Promise(r => setTimeout(r, 600));
    }
  },
};

const MODES = {
  generate: {
    help: "Create an image from a prompt. Add reference images to borrow a subject, style or layout.",
    promptLabel: "Prompt", placeholder: "A ceramic teapot on a wooden table, soft morning light…",
    imagesLabel: "Reference images (optional, up to 10)", run: "Generate", min: 0, max: 10,
  },
  edit: {
    help: "Describe a change. Optionally select a region first to confine the edit to it.",
    promptLabel: "Edit instruction", placeholder: "Change the background to a sunset beach…",
    run: "Apply edit",
  },
  controlnet: {
    help: "Guide the layout with a control signal: a pose skeleton, edges, depth, line art, scribble… Add subject references to keep a specific character's look.",
    promptLabel: "Describe the image to create", placeholder: "A knight in silver armour in a misty forest, cinematic light…",
    imagesLabel: "Subject references (optional) — keep a character's look", run: "Generate", min: 0, max: 10,
  },
  face: {
    help: "Give the person in the image on the right the face from the reference, to keep a character consistent across images. Outline the face if there are several people.",
    promptLabel: "Extra instructions (optional)", placeholder: "keep the scar over the left eye…",
    run: "Apply face",
  },
  combine: {
    help: "Put subjects from several images into one new scene. Refer to them in order (first image, second image…).",
    promptLabel: "Prompt", placeholder: "The character from the first image holding the lamp from the second image, in a forest at night…",
    imagesLabel: "Subject images (2–10, in order)", run: "Combine", min: 2, max: 10,
  },
  isolate: {
    help: "Outline a subject — a character, something it holds, any item — and everything else, including the background, is removed.",
    promptLabel: "What did you select? (optional, improves accuracy)", placeholder: "the red-haired girl / the sword she holds / the lamp on the table",
    run: "Isolate subject",
  },
  background: {
    help: "Swap the background of the image on the right for the one in another image, keeping the subject.",
    promptLabel: "Extra instructions, or what the subject is (optional)", placeholder: "the woman in the red coat / late afternoon light…",
    run: "Replace background",
  },
  upscale: {
    help: "Enlarge an image with an AI upscaler: by a factor (2×, 2.5×, 10×…) or to an exact width and/or height. Optionally let Qwen restore detail first.",
    promptLabel: "Describe the image (optional, used by Enhance)", placeholder: "a portrait of an old fisherman, weathered skin…",
    run: "Upscale",
  },
};

const EDITOR_MODES = ["edit", "isolate", "face", "background"];

const ZOOM_STEP = 1.25;
const ZOOM_MAX = 32;
const ZOOM_HINT = "Scroll to zoom; drag with the middle button (or hold Space and drag) to pan.";

const ASPECTS = { "1:1": 1, "4:3": 4 / 3, "3:4": 3 / 4, "3:2": 3 / 2, "2:3": 2 / 3, "16:9": 16 / 9, "9:16": 9 / 16 };

let models = { variants: [], default: null, loaded: null };
const workspaces = new Map();
let activeId = null;
let tabCounter = 0;

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

function blobItem(blob, name = "image.png") {
  return { blob, name, url: URL.createObjectURL(blob) };
}

function isTyping(e) {
  const t = e.target;
  return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable);
}

function lightbox(url) {
  const box = document.createElement("div");
  box.className = "lightbox";
  box.innerHTML = `<img alt="">`;
  box.firstChild.src = url;
  box.onclick = () => box.remove();
  document.body.appendChild(box);
}

// ----------------------------------------------------------------------------- workspace

class Workspace {
  constructor(id, saved = {}) {
    this.id = id;
    this.name = saved.name || `Workspace ${id}`;
    this.el = $("#workspace-template").content.firstElementChild.cloneNode(true);
    this.el.id = `ws-${id}`;
    $("#workspaces").appendChild(this.el);

    this.mode = "generate";
    this.multi = [];          // reference / subject images
    this.single = null;       // { blob, url, bitmap } for edit / isolate
    this.results = [];
    this.job = null;
    this.lastSeed = null;
    this.undoStack = [];
    this.tool = "lasso";
    this.auto = null;         // the object being auto-selected: { points, labels, base, subtract }
    this.segId = null;        // the server's id for this.single, so clicks don't re-upload it
    this.view = { z: 1, x: 0, y: 0 };  // editor zoom and pan (CSS pixels), applied as a transform

    this.mask = document.createElement("canvas");
    this.tint = document.createElement("canvas");
    this.canvas = this.r("canvas");
    this.ctx = this.canvas.getContext("2d");

    this.bind();
    this.cn = new ControlStage(this);
    this.fillVariants();
    this.restore(saved);
    this.setMode(saved.mode || "generate");
  }

  r(role) { return $(`[data-role="${role}"]`, this.el); }

  // --- persistence (settings only; images live in memory)
  settingRoles() {
    return ["prompt", "variant", "aspect", "resolution", "steps", "seed", "cfg", "num-images", "negative",
      "transparent", "preview", "mask-mode", "keep-outside", "method", "crop", "clip", "restore", "keep-canvas", "keep-hair", "keep-expression", "bg-method", "keep-subject", "cn-type", "cn-is-map", "cn-invert", "cn-hands", "cn-strength",
      "cn-auto-strength", "cn-inpaint", "cn-show-source", "auto-method",
      "up-by", "up-scale", "up-width", "up-height", "up-model", "up-enhance"];
  }
  snapshot() {
    const s = { name: this.name, mode: this.mode, settings: {} };
    for (const role of this.settingRoles()) {
      const el = this.r(role);
      s.settings[role] = el.type === "checkbox" ? el.checked : el.value;
    }
    return s;
  }
  restore(saved) {
    for (const [role, v] of Object.entries(saved.settings || {})) {
      const el = this.r(role);
      if (!el) continue;
      if (el.type === "checkbox") el.checked = !!v; else el.value = v;
      // Variant options arrive asynchronously; remember the choice until they do.
      if (role === "variant" && v) el.dataset.pending = v;
    }
  }

  // --- wiring
  bind() {
    $$(".modes button", this.el).forEach(b => b.onclick = () => { this.setMode(b.dataset.mode); persist(); });
    this.r("run").onclick = () => this.run();
    this.r("cancel").onclick = () => this.cancel();
    this.r("seed-reuse").onclick = () => { if (this.lastSeed != null) this.r("seed").value = this.lastSeed; };
    this.r("clear-results").onclick = () => { this.results = []; this.renderResults(); };
    this.r("cfg").oninput = () => this.updateVisibility();
    this.el.addEventListener("change", () => persist());
    this.r("prompt").addEventListener("input", () => persist());

    // multi image list
    const multiInput = this.r("multi-input");
    multiInput.onchange = () => { this.addMulti([...multiInput.files]); multiInput.value = ""; };
    this.dropZone(this.r("multi-drop"), files => this.addMulti(files));

    this.bindSlots();

    // upscale
    $$("[data-up-scale]", this.el).forEach(b => b.onclick = () => {
      this.r("up-scale").value = b.dataset.upScale;
      this.updateUpscaleInfo();
      persist();
    });
    for (const role of ["up-by", "up-scale", "up-width", "up-height"]) {
      this.r(role).addEventListener("input", () => this.updateVisibility());
    }

    // single image (editor)
    for (const role of ["single-input", "single-input-2"]) {
      const input = this.r(role);
      input.onchange = () => { if (input.files[0]) this.setSingle(input.files[0]); input.value = ""; };
    }
    this.dropZone(this.r("canvas-wrap"), files => files[0] && this.setSingle(files[0]));

    // editor tools
    $$("[data-tool]", this.el).forEach(b => b.onclick = () => this.setTool(b.dataset.tool));
    this.r("undo").onclick = () => this.undo();
    this.r("clear-sel").onclick = () => { this.pushUndo(); this.auto = null; this.maskCtx().clearRect(0, 0, this.mask.width, this.mask.height); this.render(); };
    this.r("zoom-in").onclick = () => this.zoomBy(ZOOM_STEP);
    this.r("zoom-out").onclick = () => this.zoomBy(1 / ZOOM_STEP);
    this.r("zoom-reset").onclick = () => this.resetView();
    this.bindCanvas();
  }

  dropZone(el, onFiles) {
    el.addEventListener("dragover", e => { e.preventDefault(); el.classList.add("over"); });
    el.addEventListener("dragleave", () => el.classList.remove("over"));
    el.addEventListener("drop", e => {
      e.preventDefault();
      el.classList.remove("over");
      const files = [...e.dataTransfer.files].filter(f => f.type.startsWith("image/"));
      if (files.length) onFiles(files);
    });
  }

  // --- named single-image slots (e.g. the face reference or the control source)
  bindSlots() {
    this.slots = {};
    $$("[data-slot]", this.el).forEach(slot => {
      const name = slot.dataset.slot;
      const input = slot.querySelector("input");
      input.onchange = () => { if (input.files[0]) this.setSlot(name, input.files[0]); input.value = ""; };
      slot.querySelector(".slot-clear").onclick = e => { e.preventDefault(); e.stopPropagation(); this.setSlot(name, null); };
      slot.querySelector("img").onclick = e => { e.preventDefault(); lightbox(e.target.src); };
      this.dropZone(slot, files => this.setSlot(name, files[0]));
    });
  }

  setSlot(name, blob) {
    const old = this.slots[name];
    if (old) URL.revokeObjectURL(old.url);
    this.slots[name] = blob ? blobItem(blob, blob.name || `${name}.png`) : null;
    if (name === "cnsource" && this.cn) this.cn.setSource(blob);
    if (name === "upscale") {
      this.upSource = null;
      if (blob) {
        const item = this.slots[name];
        createImageBitmap(blob).then(b => {
          if (this.slots.upscale === item) { this.upSource = { width: b.width, height: b.height }; this.updateUpscaleInfo(); }
          b.close();
        }).catch(() => {});
      }
      this.updateUpscaleInfo();
    }
    const slot = $(`[data-slot="${name}"]`, this.el);
    const img = slot.querySelector("img");
    img.hidden = !blob;
    if (blob) img.src = this.slots[name].url;
    slot.querySelector(".slot-empty").hidden = !!blob;
    slot.querySelector(".slot-clear").hidden = !blob;
  }

  // The slots shown for the current mode, in order.
  modeSlots() {
    return $$("[data-slot]", this.el).filter(s => !s.closest("[data-show]")?.hidden).map(s => s.dataset.slot);
  }

  usesEditor() {
    return EDITOR_MODES.includes(this.mode) || (this.mode === "controlnet" && this.r("cn-inpaint").checked);
  }

  setMode(mode) {
    this.mode = mode;
    const m = MODES[mode];
    $$(".modes button", this.el).forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
    this.r("mode-help").textContent = m.help;
    this.r("prompt-label").textContent = m.promptLabel;
    this.r("prompt").placeholder = m.placeholder;
    if (m.imagesLabel) this.r("images-label").textContent = m.imagesLabel;
    this.r("run").textContent = m.run;
    this.updateEditorHint();
    this.updateVisibility();
    this.renderThumbs();
    if (this.usesEditor()) requestAnimationFrame(() => this.render());
    if (mode === "controlnet") requestAnimationFrame(() => this.cn.sync());
  }

  updateEditorHint() {
    if (this.tool === "auto") {
      this.r("editor-hint").textContent = `Auto: click an object to select it. Shift-click adds more of the same object, Alt-click excludes a part of it. With Subtract ticked, a click removes an object from the selection. ${ZOOM_HINT}`;
      return;
    }
    this.r("editor-hint").textContent = {
      isolate: "Draw around the subject with the lasso (or paint over it). A loose outline is fine — for small items, outline just that item.",
      face: "Open the image whose face should change. Optional: outline the face (and hair, if transferring it) when there are several people.",
      background: "Open the image whose subject you want to keep. Optional: outline the subject when the image has several.",
      controlnet: "Open the image to repaint and select the region to change; the rest is kept. The repainted part follows the control map too, if there is one.",
    }[this.mode] || "Optional: select the area to change. Leave empty to edit the whole image.";
    this.r("editor-hint").textContent += ` ${ZOOM_HINT}`;
  }

  updateVisibility() {
    $$("[data-show]", this.el).forEach(el => el.hidden = !el.dataset.show.split(" ").includes(this.mode));
    $$("[data-show-cfg]", this.el).forEach(el => el.hidden = !(parseFloat(this.r("cfg").value) > 1));
    this.r("single-drop").hidden = !!this.single;
    if (this.mode === "upscale") {
      const bySize = this.r("up-by").value === "size";
      this.r("up-scale-row").hidden = bySize;
      this.r("up-size-row").hidden = !bySize;
      this.updateUpscaleInfo();
    }
    if (this.mode === "controlnet") {
      const inpaint = this.r("cn-inpaint").checked;
      $(".editor", this.el).hidden = !inpaint;
      this.r("keep-outside-row").hidden = !inpaint;
    }
  }

  // --- upscale: the output size, computed as the server does (aspect kept; width + height = fit inside)
  upscaleTarget() {
    const src = this.upSource;
    if (!src) return null;
    let k;
    if (this.r("up-by").value === "size") {
      const w = parseInt(this.r("up-width").value, 10), h = parseInt(this.r("up-height").value, 10);
      if (!(w > 0) && !(h > 0)) return null;
      k = Math.min(w > 0 ? w / src.width : Infinity, h > 0 ? h / src.height : Infinity);
    } else {
      k = parseFloat(this.r("up-scale").value);
      if (!(k > 0)) return null;
    }
    return { k, width: Math.max(1, Math.round(src.width * k)), height: Math.max(1, Math.round(src.height * k)) };
  }

  updateUpscaleInfo() {
    const info = this.r("up-info");
    const src = this.upSource;
    const t = this.upscaleTarget();
    if (!src) { info.textContent = ""; return; }
    if (!t) { info.textContent = `${src.width} × ${src.height}`; return; }
    const mp = (t.width * t.height / 1e6).toFixed(1);
    const tooBig = Math.max(t.width, t.height) > 16384 || t.k > 16;
    info.textContent = `${src.width} × ${src.height} → ${t.width} × ${t.height} (${+t.k.toFixed(3)}×, ${mp} MP)` +
      (tooBig ? " — too large: at most 16× and 16384 px per side" : "");
    info.classList.toggle("error", tooBig);
  }

  // --- image inputs
  addMulti(files) {
    const max = MODES[this.mode].max || 10;
    for (const f of files) {
      if (this.multi.length >= max) { this.setStatus(`At most ${max} images.`, true); break; }
      this.multi.push(blobItem(f, f.name));
    }
    this.renderThumbs();
    if (this.cn) this.cn.sync();
  }

  renderThumbs() {
    const box = this.r("thumbs");
    box.innerHTML = "";
    this.multi.forEach((item, i) => {
      const t = document.createElement("div");
      t.className = "thumb";
      t.innerHTML = `<img alt=""><span class="idx">${i + 1}</span><button title="Remove" aria-label="Remove">×</button>`;
      t.querySelector("img").src = item.url;
      t.querySelector("img").onclick = () => lightbox(item.url);
      t.querySelector("button").onclick = () => {
        URL.revokeObjectURL(item.url);
        this.multi.splice(i, 1);
        this.renderThumbs();
        if (this.cn) this.cn.sync();
      };
      box.appendChild(t);
    });
  }

  async setSingle(blob) {
    const bitmap = await createImageBitmap(blob);
    if (this.single) URL.revokeObjectURL(this.single.url);
    this.single = { ...blobItem(blob, blob.name || "image.png"), bitmap };
    for (const c of [this.canvas, this.mask, this.tint]) { c.width = bitmap.width; c.height = bitmap.height; }
    this.undoStack = [];
    this.auto = null;
    this.segId = null;
    this.resetView();
    this.updateVisibility();
    this.render();
  }

  // --- selection editor
  maskCtx() { return this.mask.getContext("2d", { willReadFrequently: true }); }

  setTool(tool) {
    this.tool = tool;
    $$("[data-tool]", this.el).forEach(b => b.classList.toggle("active", b.dataset.tool === tool));
    this.canvas.classList.toggle("auto", tool === "auto");
    this.canvas.classList.toggle("brush", this.isBrush());
    this.r("auto-method-row").hidden = tool !== "auto";
    this.r("brush-size-row").hidden = tool === "auto";
    this.updateEditorHint();
    this.showBrush(this.pointer);
    this.render();
  }

  isBrush() { return this.tool === "brush" || this.tool === "erase"; }

  // Outline the brush around the pointer at (clientX, clientY), so its size shows before a click.
  // A brush is brush-size screen pixels across, the same at any image size.
  showBrush(at) {
    const ring = this.r("brush-cursor");
    ring.hidden = !(at && this.single && this.isBrush());
    if (ring.hidden) return;
    const wrapEl = this.r("canvas-wrap");
    const wrap = wrapEl.getBoundingClientRect();
    ring.style.width = ring.style.height = `${this.r("brush-size").value}px`;
    ring.style.left = `${at.x - wrap.left - wrapEl.clientLeft}px`;  // relative to the padding box
    ring.style.top = `${at.y - wrap.top - wrapEl.clientTop}px`;
    ring.classList.toggle("erase", this.tool === "erase");
  }

  // Change the brush size by a factor, e.g. from the [ and ] keys.
  resizeBrush(factor) {
    const input = this.r("brush-size");
    const cur = parseInt(input.value, 10);
    let v = Math.round(cur * factor);
    if (v === cur) v += factor > 1 ? 1 : -1;
    input.value = Math.min(+input.max, Math.max(+input.min, v));
    this.showBrush(this.pointer || this.canvasCentre());
    clearTimeout(this.brushTimer);
    if (!this.pointer) this.brushTimer = setTimeout(() => this.showBrush(this.pointer), 700);
  }

  canvasCentre() {
    const rect = this.canvas.getBoundingClientRect();
    return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
  }

  pushUndo() {
    if (!this.single) return;
    this.undoStack.push(this.maskCtx().getImageData(0, 0, this.mask.width, this.mask.height));
    if (this.undoStack.length > 20) this.undoStack.shift();
  }

  undo() {
    const prev = this.undoStack.pop();
    if (prev) { this.auto = null; this.maskCtx().putImageData(prev, 0, 0); this.render(); }
  }

  // --- auto selection: the server segments the object under the click (SAM 2.1, or magic wand)
  async autoSelect(p, e) {
    if (this.autoBusy) return;
    const ctx = this.maskCtx();
    const refine = this.auto && (e.shiftKey || e.altKey);
    if (!refine) {
      this.pushUndo();
      this.auto = {
        points: [], labels: [],
        base: ctx.getImageData(0, 0, this.mask.width, this.mask.height),
        // Alt-click with nothing to refine removes the clicked object instead.
        subtract: this.r("subtract").checked || e.altKey,
      };
    }
    const obj = this.auto;
    obj.points.push([Math.round(p.x), Math.round(p.y)]);
    obj.labels.push(refine && e.altKey ? 0 : 1);
    const single = this.single;
    this.autoBusy = true;
    this.canvas.classList.add("busy");
    const first = !(models.segment && models.segment.loaded) && this.r("auto-method").value === "ai";
    this.setStatus(first ? "Auto-selecting… (the first time loads the model; it downloads ~0.9 GB once)" : "Auto-selecting…");
    this.render();
    try {
      const bitmap = await createImageBitmap(await this.segmentRequest(obj));
      if (this.single !== single || this.auto !== obj) return;  // image changed or selection reset meanwhile
      ctx.putImageData(obj.base, 0, 0);
      ctx.save();
      ctx.globalCompositeOperation = obj.subtract ? "destination-out" : "source-over";
      ctx.drawImage(bitmap, 0, 0, this.mask.width, this.mask.height);
      ctx.restore();
      this.setStatus("");
      if (first) refreshModels();
    } catch (err) {
      obj.points.pop(); obj.labels.pop();
      if (!obj.points.length && this.auto === obj) { this.auto = null; this.undoStack.pop(); }
      this.setStatus(`Auto-select failed: ${err.message}`, true);
    } finally {
      this.autoBusy = false;
      this.canvas.classList.remove("busy");
      this.render();
    }
  }

  async segmentRequest(obj) {
    const send = withImage => {
      const f = new FormData();
      if (withImage) f.append("image", this.single.blob, this.single.name);
      else f.append("image_id", this.segId);
      f.append("points", JSON.stringify(obj.points));
      f.append("labels", JSON.stringify(obj.labels));
      f.append("method", this.r("auto-method").value);
      f.append("output", "alpha");
      return fetch("/api/segment", { method: "POST", body: f });
    };
    let res = await send(!this.segId);
    if (res.status === 404 && this.segId) res = await send(true);  // the server forgot the image
    if (!res.ok) {
      let detail = null;
      try { detail = (await res.json()).detail; } catch { /* non-JSON */ }
      throw new Error(typeof detail === "string" ? detail : `HTTP ${res.status}`);
    }
    this.segId = res.headers.get("X-Image-Id") || this.segId;
    return res.blob();
  }

  toImage(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (this.canvas.width / rect.width),
      y: (e.clientY - rect.top) * (this.canvas.height / rect.height),
    };
  }

  scale() {
    const rect = this.canvas.getBoundingClientRect();
    return rect.width ? this.canvas.width / rect.width : 1;
  }

  // --- zoom and pan. The transform scales the canvas about its top-left corner, so toImage()
  // and scale(), which read the on-screen rect, keep working unchanged.
  applyView() {
    const v = this.view;
    this.canvas.style.transform = v.z === 1 ? "" : `translate(${v.x}px, ${v.y}px) scale(${v.z})`;
    this.r("zoom-reset").textContent = `${Math.round(v.z * 100)}%`;
  }

  // Keep the zoomed canvas covering its unzoomed box, so the image can't be panned out of view.
  clampView() {
    const v = this.view;
    const w = this.canvas.offsetWidth, h = this.canvas.offsetHeight;
    v.x = Math.min(0, Math.max(w * (1 - v.z), v.x));
    v.y = Math.min(0, Math.max(h * (1 - v.z), v.y));
  }

  // Zoom by a factor, keeping the image point under (clientX, clientY) in place (default: the centre).
  zoomBy(factor, clientX, clientY) {
    if (!this.single) return;
    const v = this.view;
    const rect = this.canvas.getBoundingClientRect();
    if (clientX == null) { clientX = rect.left + rect.width / 2; clientY = rect.top + rect.height / 2; }
    const z = Math.min(ZOOM_MAX, Math.max(1, v.z * factor));
    const left = rect.left - v.x, top = rect.top - v.y;  // the canvas's unzoomed position
    v.x = clientX - left - ((clientX - rect.left) / v.z) * z;
    v.y = clientY - top - ((clientY - rect.top) / v.z) * z;
    v.z = z;
    this.clampView();
    this.applyView();
    this.render();
  }

  resetView() {
    this.view = { z: 1, x: 0, y: 0 };
    this.applyView();
  }

  bindCanvas() {
    const c = this.canvas;
    let drawing = null;
    const brushRadius = () => (parseInt(this.r("brush-size").value, 10) / 2) * this.scale();
    const dab = (from, to, erase) => {
      const ctx = this.maskCtx();
      ctx.save();
      ctx.globalCompositeOperation = erase ? "destination-out" : "source-over";
      ctx.strokeStyle = ctx.fillStyle = "#fff";
      ctx.lineCap = ctx.lineJoin = "round";
      ctx.lineWidth = brushRadius() * 2;
      ctx.beginPath();
      // A zero-length stroke draws nothing, so a click without a drag paints a dot instead.
      if (from.x === to.x && from.y === to.y) { ctx.arc(to.x, to.y, brushRadius(), 0, Math.PI * 2); ctx.fill(); }
      else { ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke(); }
      ctx.restore();
    };

    // Middle-drag, or Space + drag, pans the zoomed image.
    let panning = null;
    c.addEventListener("mousedown", e => { if (e.button === 1) e.preventDefault(); });  // no autoscroll
    c.addEventListener("pointerdown", e => {
      if (!this.single || !(e.button === 1 || (e.button === 0 && this.spaceDown))) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      c.setPointerCapture(e.pointerId);
      panning = { x: e.clientX - this.view.x, y: e.clientY - this.view.y };
      c.classList.add("panning");
    });
    c.addEventListener("pointermove", e => {
      if (!panning) return;
      this.view.x = e.clientX - panning.x;
      this.view.y = e.clientY - panning.y;
      this.clampView();
      this.applyView();
    });
    const endPan = () => { panning = null; c.classList.remove("panning"); };
    c.addEventListener("pointerup", endPan);
    c.addEventListener("pointercancel", endPan);
    this.r("canvas-wrap").addEventListener("wheel", e => {
      if (!this.single) return;
      e.preventDefault();
      const dy = e.deltaY * (e.deltaMode === 1 ? 40 : e.deltaMode === 2 ? 800 : 1);  // lines / pages → pixels
      this.zoomBy(Math.exp(-dy * 0.0015), e.clientX, e.clientY);
    }, { passive: false });

    c.addEventListener("pointerdown", e => {
      if (!this.single || e.button !== 0) return;
      const p = this.toImage(e);
      if (this.tool === "auto") { e.preventDefault(); this.autoSelect(p, e); return; }
      c.setPointerCapture(e.pointerId);
      this.auto = null;
      this.pushUndo();
      drawing = { tool: this.tool, points: [p], start: p, last: p, subtract: this.r("subtract").checked };
      if (drawing.tool === "brush" || drawing.tool === "erase") dab(p, p, drawing.tool === "erase");
      this.render(drawing);
    });
    // The brush outline follows the pointer; while the size slider moves, it shows in the centre.
    c.addEventListener("pointermove", e => { this.pointer = { x: e.clientX, y: e.clientY }; this.showBrush(this.pointer); });
    c.addEventListener("pointerleave", () => { this.pointer = null; this.showBrush(null); });
    const size = this.r("brush-size");
    size.addEventListener("input", () => this.showBrush(this.canvasCentre()));
    size.addEventListener("change", () => this.showBrush(this.pointer));

    c.addEventListener("pointermove", e => {
      if (!drawing) return;
      const p = this.toImage(e);
      if (drawing.tool === "brush" || drawing.tool === "erase") dab(drawing.last, p, drawing.tool === "erase");
      else if (drawing.tool === "lasso") drawing.points.push(p);
      drawing.last = p;
      this.render(drawing);
    });
    const finish = () => {
      if (!drawing) return;
      const ctx = this.maskCtx();
      ctx.save();
      ctx.globalCompositeOperation = drawing.subtract ? "destination-out" : "source-over";
      ctx.fillStyle = "#fff";
      if (drawing.tool === "lasso" && drawing.points.length > 2) {
        ctx.beginPath();
        drawing.points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
        ctx.closePath(); ctx.fill();
      } else if (drawing.tool === "box") {
        const { start: a, last: b } = drawing;
        ctx.fillRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
      }
      ctx.restore();
      drawing = null;
      this.render();
    };
    c.addEventListener("pointerup", finish);
    c.addEventListener("pointercancel", finish);
  }

  render(drawing = null) {
    if (!this.single) return;
    const { ctx, canvas } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(this.single.bitmap, 0, 0);

    const t = this.tint.getContext("2d");
    t.globalCompositeOperation = "source-over";
    t.clearRect(0, 0, canvas.width, canvas.height);
    t.drawImage(this.mask, 0, 0);
    t.globalCompositeOperation = "source-in";
    t.fillStyle = "rgb(255, 40, 80)";
    t.fillRect(0, 0, canvas.width, canvas.height);
    ctx.globalAlpha = 0.45;
    ctx.drawImage(this.tint, 0, 0);
    ctx.globalAlpha = 1;

    if (drawing && (drawing.tool === "lasso" || drawing.tool === "box")) {
      ctx.save();
      ctx.lineWidth = Math.max(2, 2 * this.scale());
      ctx.setLineDash([8 * this.scale(), 6 * this.scale()]);
      ctx.strokeStyle = drawing.subtract ? "#00d0ff" : "#ff2850";
      ctx.beginPath();
      if (drawing.tool === "lasso") drawing.points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
      else {
        const { start: a, last: b } = drawing;
        ctx.rect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(b.x - a.x), Math.abs(b.y - a.y));
      }
      ctx.stroke();
      ctx.restore();
    }

    if (this.tool === "auto" && this.auto) {
      const r = Math.max(4, 5 * this.scale());
      ctx.save();
      ctx.lineWidth = Math.max(1.5, 1.5 * this.scale());
      ctx.strokeStyle = "#fff";
      this.auto.points.forEach(([x, y], i) => {
        ctx.fillStyle = this.auto.labels[i] ? "#22c55e" : "#ef4444";
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      });
      ctx.restore();
    }
  }

  hasSelection() {
    if (!this.single) return false;
    const data = this.maskCtx().getImageData(0, 0, this.mask.width, this.mask.height).data;
    for (let i = 3; i < data.length; i += 4) if (data[i] > 127) return true;
    return false;
  }

  maskBlob() {
    return new Promise(resolve => this.mask.toBlob(resolve, "image/png"));
  }

  // --- run
  sizeFields(form) {
    const aspect = this.r("aspect").value;
    const res = parseInt(this.r("resolution").value, 10);
    form.append("resolution", res);
    if (aspect !== "auto" && !["isolate", "face", "background", "upscale"].includes(this.mode)) {
      const a = ASPECTS[aspect];
      form.append("width", Math.round(res * Math.sqrt(a) / 32) * 32);
      form.append("height", Math.round(res / Math.sqrt(a) / 32) * 32);
    }
  }

  async buildForm() {
    const f = new FormData();
    const prompt = this.r("prompt").value.trim();
    const add = (k, v) => { if (v !== "" && v != null) f.append(k, v); };
    add("variant", this.r("variant").value);
    add("steps", this.r("steps").value);
    add("seed", this.r("seed").value);
    add("cfg", this.r("cfg").value);
    add("num_images", this.r("num-images").value);
    add("preview", this.r("preview").value);
    if (parseFloat(this.r("cfg").value) > 1) add("negative_prompt", this.r("negative").value);
    this.sizeFields(f);

    if (this.mode === "generate" || this.mode === "combine") {
      const m = MODES[this.mode];
      if (!prompt) throw new Error("Enter a prompt.");
      if (this.multi.length < m.min) throw new Error(`Add at least ${m.min} images.`);
      f.append("prompt", prompt);
      f.append("transparent", this.r("transparent").checked);
      const field = this.mode === "generate" ? "references" : "images";
      this.multi.forEach(item => f.append(field, item.blob, item.name));
    } else if (this.mode === "controlnet") {
      if (!prompt) throw new Error("Describe the image to create.");
      f.append("prompt", prompt);
      f.append("control_type", this.r("cn-type").value);
      f.append("transparent", this.r("transparent").checked);
      if (!this.r("cn-auto-strength").checked) f.append("strength", this.r("cn-strength").value);
      this.multi.forEach(item => f.append("references", item.blob, item.name));
      const hasControl = this.cn.appendTo(f);
      const inpaint = this.r("cn-inpaint").checked;
      if (inpaint) {
        if (!this.single) throw new Error("Open the image to repaint (below the control map).");
        if (!this.hasSelection()) throw new Error("Select the region to repaint.");
        f.append("image", this.single.blob, this.single.name);
        f.append("mask", await this.maskBlob(), "mask.png");
        f.append("keep_outside", this.r("keep-outside").checked);
      }
      if (!hasControl && !inpaint) throw new Error("Add a control source, draw a pose, or turn on inpainting.");
    } else if (this.mode === "upscale") {
      if (!this.slots.upscale) throw new Error("Add the image to upscale.");
      f.append("image", this.slots.upscale.blob, this.slots.upscale.name);
      if (this.r("up-by").value === "size") {
        const w = this.r("up-width").value, h = this.r("up-height").value;
        if (!w && !h) throw new Error("Enter a target width and/or height.");
        add("width", w);
        add("height", h);
      } else {
        const k = parseFloat(this.r("up-scale").value);
        if (!(k > 0)) throw new Error("Enter a scale factor, e.g. 2.5.");
        f.append("scale", k);
      }
      f.append("model", this.r("up-model").value);
      f.append("enhance", this.r("up-enhance").checked);
      f.append("prompt", prompt);
    } else {
      if (!this.single) throw new Error("Open an image first.");
      f.append("image", this.single.blob, this.single.name);
      const selected = this.hasSelection();
      if (selected) f.append("mask", await this.maskBlob(), "mask.png");
      if (this.mode === "edit") {
        if (!prompt) throw new Error("Describe the edit.");
        f.append("prompt", prompt);
        f.append("transparent", this.r("transparent").checked);
        f.append("mask_mode", this.r("mask-mode").value);
        f.append("keep_outside", this.r("keep-outside").checked);
      } else if (this.mode === "background") {
        if (!this.slots.background) throw new Error("Add an image with the background you want.");
        f.append("background", this.slots.background.blob, this.slots.background.name);
        f.append("prompt", prompt);
        f.append("method", this.r("bg-method").value);
        f.append("keep_subject", this.r("keep-subject").checked);
      } else if (this.mode === "face") {
        if (!this.slots.face) throw new Error("Add an image showing the face to use.");
        f.append("face", this.slots.face.blob, this.slots.face.name);
        f.append("prompt", prompt);
        f.append("keep_hair", this.r("keep-hair").checked);
        f.append("keep_expression", this.r("keep-expression").checked);
        f.append("keep_outside", this.r("keep-outside").checked);
      } else {
        if (!selected) throw new Error("Outline the subject to isolate first.");
        f.append("description", prompt);
        f.append("method", this.r("method").value);
        f.append("crop", this.r("crop").checked);
        f.append("clip", this.r("clip").checked);
        f.append("restore_pixels", this.r("restore").checked);
        f.append("keep_canvas", this.r("keep-canvas").checked);
      }
    }
    return f;
  }

  setStatus(text, error = false) {
    const s = this.r("status");
    s.textContent = text;
    s.classList.toggle("error", error);
  }

  setBusy(busy) {
    this.r("run").disabled = busy;
    this.r("cancel").hidden = !busy;
    this.r("progress").hidden = !busy;
    renderTabs();
  }

  async run() {
    let form;
    try { form = await this.buildForm(); } catch (e) { this.setStatus(e.message, true); return; }
    this.setBusy(true);
    this.setStatus("Submitting…");
    const bar = $(".bar", this.r("progress"));
    bar.style.width = "0";
    const mode = this.mode;
    const live = this.r("preview").value !== "off" && !(mode === "isolate" && this.r("method").value === "mask")
      && !(mode === "upscale" && !this.r("up-enhance").checked);
    try {
      const created = await api.post(`/api/${mode}`, form);
      this.job = created;
      const job = await api.waitJob(created.id, j => {
        this.job = j;
        const { step, total } = j.progress;
        this.r("progress").classList.toggle("indeterminate", !total);
        if (total) bar.style.width = `${(100 * step) / total}%`;
        if (live) this.showLive(j);
        if (j.status === "queued") this.setStatus(`Queued (position ${j.queue_position})…`);
        else if (j.status === "running") this.setStatus(total ? `${j.message || "Sampling"} — step ${step}/${total}` : (j.message || "Working…"));
      });
      if (job.status === "done") {
        this.lastSeed = job.seed;
        await this.addResults(job);
        const secs = job.finished && job.started ? ` in ${(job.finished - job.started).toFixed(1)}s` : "";
        this.setStatus(`Done${secs}${job.seed >= 0 ? ` · seed ${job.seed}` : ""}`);
      } else if (job.status === "cancelled") {
        this.setStatus("Cancelled.");
      } else {
        this.setStatus(job.error || "Failed.", true);
      }
    } catch (e) {
      this.setStatus(e.message, true);
    } finally {
      this.hideLive();
      this.job = null;
      this.setBusy(false);
      refreshHealth();
    }
  }

  async cancel() {
    if (this.job) {
      try { await api.del(`/api/jobs/${this.job.id}`); } catch { /* already finished */ }
      this.setStatus("Cancelling…");
    }
  }

  // --- live preview
  showLive(job) {
    if (!this.live) {
      const card = document.createElement("div");
      card.className = "result live";
      card.innerHTML = `<div class="img"><img alt="Live preview" hidden><span class="live-wait">Waiting for the first step…</span></div><div class="meta"></div>`;
      card.querySelector("img").onclick = e => lightbox(e.target.src);
      this.live = { card, url: null };
      this.renderResults();
    }
    const { card } = this.live;
    if (job.preview && job.preview !== this.live.url) {
      this.live.url = job.preview;
      const img = card.querySelector("img");
      img.src = job.preview;
      img.hidden = false;
      card.querySelector(".live-wait").hidden = true;
    }
    const total = job.progress.total;
    card.querySelector(".meta").textContent = job.status === "queued"
      ? "Queued…"
      : job.preview_step ? `Live preview · step ${job.preview_step}/${total}` : (job.message || "Starting…");
  }

  hideLive() {
    if (!this.live) return;
    this.live.card.remove();
    this.live = null;
    if (!this.results.length) this.renderResults();
  }

  // --- results
  async addResults(job) {
    const items = await Promise.all(job.images.map(async (url, i) => {
      const blob = await (await fetch(url)).blob();
      return { ...blobItem(blob, `${job.kind}-${job.id}-${i}.png`), api: url, job };
    }));
    this.results.unshift(...items);
    this.renderResults();
  }

  renderResults() {
    const box = this.r("results");
    box.innerHTML = "";
    if (this.live) box.appendChild(this.live.card);
    if (!this.results.length) {
      if (!this.live) box.innerHTML = `<p class="empty">Results appear here.</p>`;
      return;
    }
    for (const item of this.results) {
      const card = document.createElement("div");
      card.className = "result";
      card.innerHTML = `
        <div class="img"><img alt=""></div>
        <div class="meta"></div>
        <div class="actions">
          <a download>Download</a>
          <button data-a="edit">Edit</button>
          <button data-a="isolate">Isolate</button>
          <button data-a="upscale">Upscale</button>
          <button data-a="ref">Use as input</button>
          <button data-a="cmap" hidden>Control map</button>
          <button data-a="tab">New tab</button>
        </div>`;
      const img = card.querySelector("img");
      img.src = item.url;
      img.onclick = () => lightbox(item.url);
      const j = item.job;
      const size = j.kind === "upscale" && j.info && j.info.size ? ` · ${j.info.size[0]} × ${j.info.size[1]}` : "";
      card.querySelector(".meta").textContent = `${j.kind}${j.variant ? ` · ${j.variant}` : ""}${size}${j.seed >= 0 ? ` · seed ${j.seed}` : ""}`;
      card.querySelector(".meta").title = j.prompt || "";
      const a = card.querySelector("a");
      a.href = item.url;
      a.download = item.name;
      card.querySelector('[data-a="edit"]').onclick = () => { this.setMode("edit"); this.setSingle(item.blob); };
      card.querySelector('[data-a="isolate"]').onclick = () => { this.setMode("isolate"); this.setSingle(item.blob); };
      card.querySelector('[data-a="upscale"]').onclick = () => { this.setMode("upscale"); this.setSlot("upscale", item.blob); persist(); };
      card.querySelector('[data-a="ref"]').onclick = () => {
        if (this.usesEditor()) this.setSingle(item.blob);
        else if (this.mode === "controlnet") this.setSlot("cnsource", item.blob);
        else if (this.mode === "upscale") this.setSlot("upscale", item.blob);
        else this.addMulti([new File([item.blob], item.name, { type: "image/png" })]);
      };
      const cmap = j.extras && j.extras.control;
      if (cmap) {
        const btn = card.querySelector('[data-a="cmap"]');
        btn.hidden = false;
        btn.onclick = () => lightbox(cmap);
      }
      card.querySelector('[data-a="tab"]').onclick = () => {
        const ws = createTab({ mode: "edit" });
        ws.setSingle(item.blob);
      };
      box.appendChild(card);
    }
  }

  fillVariants() {
    const sel = this.r("variant");
    const current = sel.value;
    sel.innerHTML = "";
    const def = document.createElement("option");
    def.value = "";
    def.textContent = `Server default${models.default ? ` (${models.default})` : ""}`;
    sel.appendChild(def);
    for (const v of models.variants) {
      const o = document.createElement("option");
      o.value = v.name;
      const state = v.loaded ? " — loaded" : v.downloaded ? "" : ` — downloads ~${v.approx_download_gb} GB`;
      o.textContent = `${v.label}${state}`;
      sel.appendChild(o);
    }
    const want = current || sel.dataset.pending || "";
    sel.value = want;
    if (sel.value !== want) sel.value = "";
    if (models.variants.length) delete sel.dataset.pending;
  }

  destroy() {
    if (this.job) this.cancel();
    const slots = Object.values(this.slots || {});
    for (const item of [...this.multi, ...this.results, ...slots, this.single].filter(Boolean)) URL.revokeObjectURL(item.url);
    this.el.remove();
  }
}

// ----------------------------------------------------------------------------- tabs

function createTab(saved = {}, activate = true) {
  const id = ++tabCounter;
  const ws = new Workspace(id, saved);
  workspaces.set(id, ws);
  if (activate) activateTab(id);
  persist();
  return ws;
}

function closeTab(id) {
  const ws = workspaces.get(id);
  if (!ws) return;
  if (ws.job && !confirm(`“${ws.name}” has a job running. Close it and cancel the job?`)) return;
  const ids = [...workspaces.keys()];
  const idx = ids.indexOf(id);
  ws.destroy();
  workspaces.delete(id);
  if (!workspaces.size) createTab();
  else if (activeId === id) activateTab([...workspaces.keys()][Math.max(0, idx - 1)]);
  renderTabs();
  persist();
}

function activateTab(id) {
  activeId = id;
  for (const [wid, ws] of workspaces) ws.el.classList.toggle("active", wid === id);
  renderTabs();
  const ws = workspaces.get(id);
  if (ws) requestAnimationFrame(() => ws.render());
  persist();
}

function renderTabs() {
  const bar = $("#tabs");
  bar.innerHTML = "";
  for (const [id, ws] of workspaces) {
    const tab = document.createElement("div");
    tab.className = "tab" + (id === activeId ? " active" : "");
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-selected", id === activeId);
    tab.title = "Double-click to rename";
    tab.innerHTML = `${ws.job ? '<span class="busy" title="Working"></span>' : ""}<span class="name"></span><button class="close" title="Close tab" aria-label="Close tab">×</button>`;
    tab.querySelector(".name").textContent = ws.name;
    tab.onclick = e => { if (id !== activeId && !e.target.classList.contains("close")) activateTab(id); };
    tab.onauxclick = e => { if (e.button === 1) closeTab(id); };
    tab.querySelector(".close").onclick = () => closeTab(id);
    tab.ondblclick = () => {
      const input = document.createElement("input");
      input.value = ws.name;
      const name = tab.querySelector(".name");
      name.replaceWith(input);
      input.focus(); input.select();
      const done = () => { ws.name = input.value.trim() || ws.name; renderTabs(); persist(); };
      input.onblur = done;
      input.onkeydown = e => { if (e.key === "Enter") input.blur(); if (e.key === "Escape") { input.value = ws.name; input.blur(); } };
    };
    bar.appendChild(tab);
  }
}

const STORAGE_KEY = "poly-qwen-image.tabs";
let persistTimer = null;
function persist() {
  clearTimeout(persistTimer);
  persistTimer = setTimeout(() => {
    try {
      const ids = [...workspaces.keys()];
      const state = { tabs: [...workspaces.values()].map(ws => ws.snapshot()), active: ids.indexOf(activeId) };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch { /* storage unavailable */ }
  }, 300);
}

function restoreTabs() {
  let state = null;
  try { state = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch { /* ignore */ }
  if (state && state.tabs && state.tabs.length) {
    state.tabs.forEach(t => createTab(t, false));
    const ids = [...workspaces.keys()];
    activateTab(ids[Math.min(Math.max(0, state.active), ids.length - 1)]);
  } else {
    createTab();
  }
}

// ----------------------------------------------------------------------------- models

async function refreshModels() {
  try {
    models = await api.get("/api/models");
  } catch (e) {
    $("#models-list").textContent = `Could not load models: ${e.message}`;
    return;
  }
  for (const ws of workspaces.values()) ws.fillVariants();
  renderModels();
  updatePill();
}

function updatePill(text) {
  $("#model-status").textContent = text || (models.loaded ? `Loaded: ${models.loaded}` : `No model loaded · default ${models.default || "?"}`);
  $("#model-dot").className = "dot" + (text ? " busy" : models.loaded ? " ok" : "");
}

async function refreshHealth() {
  try {
    const h = await api.get("/api/health");
    if (h.loaded !== models.loaded) await refreshModels();
  } catch { /* server unreachable */ }
}

function renderModels() {
  const list = $("#models-list");
  list.innerHTML = "";
  for (const v of models.variants) {
    const row = document.createElement("div");
    row.className = "model-row" + (v.loaded ? " loaded" : "");
    row.innerHTML = `
      <div class="title"></div>
      <div class="desc"></div>
      <div class="btns">
        <button data-a="download" ${v.downloaded ? "hidden" : ""}>Download</button>
        <button data-a="load" class="${v.loaded ? "" : "primary"}" ${v.loaded ? "disabled" : ""}>${v.loaded ? "Loaded" : "Load"}</button>
      </div>`;
    const title = row.querySelector(".title");
    title.textContent = v.label;
    const badge = document.createElement("span");
    badge.className = "badge" + (v.downloaded ? " ok" : "");
    badge.textContent = v.downloaded ? "downloaded" : `~${v.approx_download_gb} GB`;
    title.appendChild(badge);
    if (v.name === models.default) {
      const d = document.createElement("span");
      d.className = "badge"; d.textContent = "default";
      title.appendChild(d);
    }
    row.querySelector(".desc").textContent = `${v.name} — ${v.description}`;
    const runJob = async (path, btn) => {
      const form = new FormData();
      form.append("variant", v.name);
      $$("button", row).forEach(b => b.disabled = true);
      try {
        const job = await api.post(path, form);
        const final = await api.waitJob(job.id, j => { btn.textContent = j.status === "queued" ? "Queued…" : (j.message || "Working…").slice(0, 40); updatePill(`${v.name}: ${j.message || j.status}`); });
        if (final.status === "error") alert(final.error);
      } catch (e) { alert(e.message); }
      await refreshModels();
    };
    row.querySelector('[data-a="download"]').onclick = e => runJob("/api/models/download", e.target);
    row.querySelector('[data-a="load"]').onclick = e => runJob("/api/models/load", e.target);
    list.appendChild(row);
  }
  if (models.controlnet) {
    addonRow(list, "ControlNet-Union add-on", models.controlnet, "/api/controlnet/download",
      "pose, edges, depth, line art, scribble and inpainting control for the ControlNet tab. Works with every variant; downloads on first use.");
  }
  if (models.segment && models.segment.repo !== "mock") {
    addonRow(list, "Auto-select (SAM 2.1)", models.segment, "/api/segment/download",
      "click-to-select for the Auto selection tool. Runs alongside any variant; downloads on first use.");
  }
}

function addonRow(list, label, info, downloadPath, desc) {
  const row = document.createElement("div");
  row.className = "model-row" + (info.loaded ? " loaded" : "");
  row.innerHTML = `
    <div class="title"></div>
    <div class="desc"></div>
    <div class="btns"><button data-a="download" ${info.downloaded ? "hidden" : ""}>Download</button></div>`;
  row.querySelector(".title").textContent = label;
  const badge = document.createElement("span");
  badge.className = "badge" + (info.downloaded ? " ok" : "");
  badge.textContent = info.loaded ? "loaded" : info.downloaded ? "downloaded" : `~${info.approx_download_gb} GB`;
  row.querySelector(".title").appendChild(badge);
  row.querySelector(".desc").textContent = `${info.repo} — ${desc}`;
  row.querySelector('[data-a="download"]').onclick = async e => {
    e.target.disabled = true;
    try {
      const job = await api.post(downloadPath, new FormData());
      const final = await api.waitJob(job.id, j => { e.target.textContent = (j.message || j.status).slice(0, 40); });
      if (final.status === "error") alert(final.error);
    } catch (err) { alert(err.message); }
    await refreshModels();
  };
  list.appendChild(row);
}

// ----------------------------------------------------------------------------- global events

$("#new-tab").onclick = () => createTab();
$("#model-pill").onclick = () => { $("#models-dialog").showModal(); refreshModels(); };

document.addEventListener("keydown", e => {
  const ws = workspaces.get(activeId);
  if (e.ctrlKey && e.altKey && e.key.toLowerCase() === "t") { e.preventDefault(); createTab(); return; }
  if (e.ctrlKey && e.altKey && e.key.toLowerCase() === "w") { e.preventDefault(); closeTab(activeId); return; }
  if (!ws || isTyping(e)) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); ws.undo(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  if (!ws.usesEditor()) return;
  const tools = { a: "auto", l: "lasso", r: "box", b: "brush", e: "erase" };
  if (tools[e.key.toLowerCase()]) ws.setTool(tools[e.key.toLowerCase()]);
  else if ((e.key === "[" || e.key === "]") && ws.isBrush()) ws.resizeBrush(e.key === "]" ? 1.15 : 1 / 1.15);
  else if (e.key === "+" || e.key === "=") ws.zoomBy(ZOOM_STEP);
  else if (e.key === "-") ws.zoomBy(1 / ZOOM_STEP);
  else if (e.key === "0") ws.resetView();
  else if (e.code === "Space" && ws.single) {
    e.preventDefault();  // no page scroll
    ws.spaceDown = true;
    ws.canvas.classList.add("pan");
  }
});

document.addEventListener("keyup", e => {
  if (e.code !== "Space") return;
  for (const ws of workspaces.values()) { ws.spaceDown = false; ws.canvas.classList.remove("pan"); }
});

document.addEventListener("paste", e => {
  const ws = workspaces.get(activeId);
  if (!ws || isTyping(e)) return;
  const files = [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith("image/"));
  if (!files.length) return;
  e.preventDefault();
  // Fill an empty slot first, then the editor, then the image list.
  const slots = ws.modeSlots();
  const empty = slots.find(n => !ws.slots[n]);
  if (empty) ws.setSlot(empty, files[0]);
  else if (ws.usesEditor()) ws.setSingle(files[0]);
  else if (ws.mode === "controlnet") ws.addMulti(files);
  else if (slots.length) ws.setSlot(slots[slots.length - 1], files[0]);
  else ws.addMulti(files);
});

window.addEventListener("resize", () => {
  const ws = workspaces.get(activeId);
  if (ws) { ws.clampView(); ws.applyView(); ws.render(); ws.cn.render(); }
});

restoreTabs();
refreshModels();
setInterval(refreshHealth, 5000);
