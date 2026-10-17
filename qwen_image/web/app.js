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
  pose: {
    help: "Make the subject of one image take the pose from another. The pose reference can be a character, a photo, or a simple stick figure.",
    promptLabel: "Extra instructions (optional)", placeholder: "keep the sword in the right hand…",
    run: "Apply pose",
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
};

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

    this.mask = document.createElement("canvas");
    this.tint = document.createElement("canvas");
    this.canvas = this.r("canvas");
    this.ctx = this.canvas.getContext("2d");

    this.bind();
    this.fillVariants();
    this.restore(saved);
    this.setMode(saved.mode || "generate");
  }

  r(role) { return $(`[data-role="${role}"]`, this.el); }

  // --- persistence (settings only; images live in memory)
  settingRoles() {
    return ["prompt", "variant", "aspect", "resolution", "steps", "seed", "cfg", "num-images", "negative",
      "transparent", "preview", "mask-mode", "keep-outside", "method", "crop", "clip", "restore", "keep-canvas", "pose-type", "keep-background"];
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

    // single image (editor)
    for (const role of ["single-input", "single-input-2"]) {
      const input = this.r(role);
      input.onchange = () => { if (input.files[0]) this.setSingle(input.files[0]); input.value = ""; };
    }
    this.dropZone(this.r("canvas-wrap"), files => files[0] && this.setSingle(files[0]));

    // editor tools
    $$("[data-tool]", this.el).forEach(b => b.onclick = () => this.setTool(b.dataset.tool));
    this.r("undo").onclick = () => this.undo();
    this.r("clear-sel").onclick = () => { this.pushUndo(); this.maskCtx().clearRect(0, 0, this.mask.width, this.mask.height); this.render(); };
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

  // --- named single-image slots (e.g. pose reference + subject)
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

  setMode(mode) {
    this.mode = mode;
    const m = MODES[mode];
    $$(".modes button", this.el).forEach(b => b.classList.toggle("active", b.dataset.mode === mode));
    this.r("mode-help").textContent = m.help;
    this.r("prompt-label").textContent = m.promptLabel;
    this.r("prompt").placeholder = m.placeholder;
    if (m.imagesLabel) this.r("images-label").textContent = m.imagesLabel;
    this.r("run").textContent = m.run;
    this.r("editor-hint").textContent = mode === "isolate"
      ? "Draw around the subject with the lasso (or paint over it). A loose outline is fine — for small items, outline just that item."
      : "Optional: select the area to change. Leave empty to edit the whole image.";
    this.updateVisibility();
    this.renderThumbs();
    if (mode === "edit" || mode === "isolate") requestAnimationFrame(() => this.render());
  }

  updateVisibility() {
    $$("[data-show]", this.el).forEach(el => el.hidden = !el.dataset.show.split(" ").includes(this.mode));
    $$("[data-show-cfg]", this.el).forEach(el => el.hidden = !(parseFloat(this.r("cfg").value) > 1));
    this.r("single-drop").hidden = !!this.single;
  }

  // --- image inputs
  addMulti(files) {
    const max = MODES[this.mode].max || 10;
    for (const f of files) {
      if (this.multi.length >= max) { this.setStatus(`At most ${max} images.`, true); break; }
      this.multi.push(blobItem(f, f.name));
    }
    this.renderThumbs();
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
      t.querySelector("button").onclick = () => { URL.revokeObjectURL(item.url); this.multi.splice(i, 1); this.renderThumbs(); };
      box.appendChild(t);
    });
  }

  async setSingle(blob) {
    const bitmap = await createImageBitmap(blob);
    if (this.single) URL.revokeObjectURL(this.single.url);
    this.single = { ...blobItem(blob, blob.name || "image.png"), bitmap };
    for (const c of [this.canvas, this.mask, this.tint]) { c.width = bitmap.width; c.height = bitmap.height; }
    this.undoStack = [];
    this.updateVisibility();
    this.render();
  }

  // --- selection editor
  maskCtx() { return this.mask.getContext("2d", { willReadFrequently: true }); }

  setTool(tool) {
    this.tool = tool;
    $$("[data-tool]", this.el).forEach(b => b.classList.toggle("active", b.dataset.tool === tool));
  }

  pushUndo() {
    if (!this.single) return;
    this.undoStack.push(this.maskCtx().getImageData(0, 0, this.mask.width, this.mask.height));
    if (this.undoStack.length > 20) this.undoStack.shift();
  }

  undo() {
    const prev = this.undoStack.pop();
    if (prev) { this.maskCtx().putImageData(prev, 0, 0); this.render(); }
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
      ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y); ctx.stroke();
      ctx.restore();
    };

    c.addEventListener("pointerdown", e => {
      if (!this.single || e.button !== 0) return;
      c.setPointerCapture(e.pointerId);
      const p = this.toImage(e);
      this.pushUndo();
      drawing = { tool: this.tool, points: [p], start: p, last: p, subtract: this.r("subtract").checked };
      if (drawing.tool === "brush" || drawing.tool === "erase") dab(p, p, drawing.tool === "erase");
      this.render(drawing);
    });
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
    if (aspect !== "auto" && this.mode !== "isolate") {
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
    } else if (this.mode === "pose") {
      if (!this.slots.pose) throw new Error("Add a pose reference image.");
      if (!this.slots.subject) throw new Error("Add the image whose subject should be re-posed.");
      f.append("pose", this.slots.pose.blob, this.slots.pose.name);
      f.append("image", this.slots.subject.blob, this.slots.subject.name);
      f.append("prompt", prompt);
      f.append("pose_type", this.r("pose-type").value);
      f.append("keep_background", this.r("keep-background").checked);
      f.append("transparent", this.r("transparent").checked);
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
    const live = this.r("preview").value !== "off" && !(mode === "isolate" && this.r("method").value === "mask");
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
          <button data-a="ref">Use as input</button>
          <button data-a="tab">New tab</button>
        </div>`;
      const img = card.querySelector("img");
      img.src = item.url;
      img.onclick = () => lightbox(item.url);
      const j = item.job;
      card.querySelector(".meta").textContent = `${j.kind} · ${j.variant || ""}${j.seed >= 0 ? ` · seed ${j.seed}` : ""}`;
      card.querySelector(".meta").title = j.prompt || "";
      const a = card.querySelector("a");
      a.href = item.url;
      a.download = item.name;
      card.querySelector('[data-a="edit"]').onclick = () => { this.setMode("edit"); this.setSingle(item.blob); };
      card.querySelector('[data-a="isolate"]').onclick = () => { this.setMode("isolate"); this.setSingle(item.blob); };
      card.querySelector('[data-a="ref"]').onclick = () => {
        if (this.mode === "edit" || this.mode === "isolate") this.setSingle(item.blob);
        else if (this.mode === "pose") this.setSlot("subject", item.blob);
        else this.addMulti([new File([item.blob], item.name, { type: "image/png" })]);
      };
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

const STORAGE_KEY = "qwen-image-studio.tabs";
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
  const tools = { l: "lasso", r: "box", b: "brush", e: "erase" };
  if (tools[e.key.toLowerCase()] && (ws.mode === "edit" || ws.mode === "isolate")) ws.setTool(tools[e.key.toLowerCase()]);
});

document.addEventListener("paste", e => {
  const ws = workspaces.get(activeId);
  if (!ws || isTyping(e)) return;
  const files = [...(e.clipboardData?.files || [])].filter(f => f.type.startsWith("image/"));
  if (!files.length) return;
  e.preventDefault();
  const slots = ws.modeSlots();
  if (slots.length) ws.setSlot(slots.find(n => !ws.slots[n]) || slots[slots.length - 1], files[0]);
  else if (ws.mode === "edit" || ws.mode === "isolate") ws.setSingle(files[0]);
  else ws.addMulti(files);
});

window.addEventListener("resize", () => workspaces.get(activeId)?.render());

restoreTabs();
refreshModels();
setInterval(refreshHealth, 5000);
