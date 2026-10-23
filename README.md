# poly-qwen-image

A custom front end for [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1), built on PyTorch, diffusers and FastAPI. It has four core operations, and each one is available from the Python API, the HTTP API, the web UI and the CLI:

| Operation | What it does |
|---|---|
| **generate** | Text-to-image. You can also pass reference images (up to 10) to borrow a subject, style or layout. Transparent (RGBA) output is supported. |
| **edit** | Edits an image from an instruction. An optional selection limits the edit to one region, and pixels outside that region are kept by default. |
| **controlnet** | Generates guided by a control signal: a pose skeleton, canny edges, depth, line art, soft edges, straight lines, scribble or grayscale. The signal can be extracted from any image or drawn by hand; the pose skeleton is editable. Optional subject references keep a specific character's look, so a character can be re-posed. Optional inpainting repaints only a masked region. |
| **face** | Gives the subject of an image the face from a reference image, to keep characters consistent across images. Hair and expression are kept by default; either can be transferred too. An optional selection picks one face among several. |
| **combine** | Combines subjects from 2–10 images into one new image. |
| **isolate** | Cuts a selected subject out of the image. The model removes the background and everything else and returns an RGBA cutout. This works on sub-elements too, such as an item a character holds or one object in the background. |
| **background** | Replaces an image's background with the one from another image and keeps the subject. The model can repaint the scene behind the subject with matching lighting, or the subject can be cut out and pasted over the image. |
| **upscale** | Enlarges an image with an AI upscaler, by a factor (2×, 2.5×, 10×, up to 16×) or to a width and/or height (up to 16384 px). An optional Qwen pass restores detail first. |

You can choose between several model variants, including quantized ones. Each variant downloads from the Hugging Face Hub the first time it is used.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate elsewhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
```

`QwenImage21Pipeline` currently ships only on diffusers `main`, so `pyproject.toml` installs diffusers from git.

## Docker (Linux server)

`docker-compose.yml` runs the app on an NVIDIA GPU behind [Caddy](https://caddyserver.com), which serves it over HTTPS with its own certificate. The host needs Docker, the NVIDIA driver (570 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). On Ubuntu or Debian, `sudo ./NVIDIA.sh` installs all three and checks that a container can see the GPU.

```bash
cp .env.example .env          # set DOMAIN, and optionally the variant and serve flags
docker compose up -d --build
docker compose logs -f app
```

- **Certificate:** if `DOMAIN` is a public name whose DNS points at the server, Caddy gets a Let's Encrypt certificate automatically and renews it. Ports 80 and 443 must be reachable for that. For `localhost`, an IP address or a LAN-only name, Caddy issues a certificate from its own local CA, and browsers warn until you trust that CA. To use a certificate you already have, put it in `deploy/certs/` and set `CADDY_TLS=tls /certs/fullchain.pem /certs/privkey.pem`.
- **Only Caddy is exposed.** The app listens on port 7860 inside the compose network only.
- **Password:** the app has no login of its own. Copy `deploy/conf.d/auth.caddy.example` to `deploy/conf.d/auth.caddy` and fill in a hash, which turns on HTTP basic auth in front of everything. The example file explains how to make the hash.
- **Storage:** weights download on first use into the `models` volume. Generated images go to the `outputs` volume. Certificates live in `caddy_data`; keep that volume, so certificates aren't re-issued.
- **Settings:** `QWEN_IMAGE_VARIANT` sets the default variant. `QWEN_IMAGE_SERVE_ARGS` passes extra `qwen-image serve` flags, for example `--preload --offload model`. `HF_TOKEN` speeds up downloads.

The image runs on its own too: `docker build -t poly-qwen-image .`, then `docker run --gpus all -p 7860:7860 -v qwen-cache:/data/cache -v qwen-outputs:/data/outputs poly-qwen-image`.

## Model variants

```bash
qwen-image models list               # shows what is already downloaded
qwen-image models download gguf-q4_k_m
```

| Variant | Denoiser | Text encoder | Notes |
|---|---|---|---|
| `bf16` | official BF16 | BF16 | Best quality. Uses CPU offload on GPUs under 40 GB. |
| `turbo-bf16` | official Turbo | BF16 | Fixed 8-step schedule, no CFG. |
| `nf4`, `int8`, `turbo-nf4` | bitsandbytes 4/8-bit, quantized on load | NF4 | Downloads the official checkpoint. |
| `gguf-q8_0` … `gguf-q2_k` | [unsloth GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF) | NF4 | Smallest downloads. **`gguf-q4_k_m` is the default.** |

To change the default, set `QWEN_IMAGE_VARIANT` or pass `--variant`. Offloading is controlled by `--offload auto|none|model|sequential`; `auto` keeps the model on the GPU only when the GPU has at least 40 GB. Weights go in the standard HF cache (`HF_HOME`).

## Web UI and HTTP API

```bash
qwen-image serve --port 7860 [--variant gguf-q4_k_m] [--preload]
```

Open http://127.0.0.1:7860. Each **tab** is an independent workspace with all the controls. Use **+** or Ctrl+Alt+T to open a tab, × or middle-click to close one, and double-click a tab to rename it. Tab settings are kept across reloads. In Edit and Isolate, use the auto, lasso, box or brush tools to select a region (A/L/R/B/E, Ctrl+Z to undo). The brush and eraser show their outline around the pointer before you click; change their size with the Size slider or the [ and ] keys. To refine a selection, zoom in with the mouse wheel, the −/+ buttons or the −/+ keys (0 fits the image again), and pan with the middle mouse button or Space + drag. With **Auto**, click an object and AI selects it (see [Auto selection](#auto-selection)). You can paste images with Ctrl+V. The UI uses only the HTTP API below. Interactive API docs are at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/generate` | `prompt`, optional `references[]` files |
| POST | `/api/segment` | Auto selection: `image` (or `image_id`), `points` JSON `[[x, y], ...]`, optional `labels` (1 = include, 0 = exclude), `box`, `method` (`ai`/`color`), `output` (`mask`/`alpha`). Returns the mask as a PNG |
| POST | `/api/segment/download` | fetch the auto-select weights (0.9 GB) |
| POST | `/api/edit` | `image`, `prompt`, optional `mask` file or `selection` JSON, `mask_mode`, `keep_outside` |
| POST | `/api/controlnet` | `prompt`, `control_type`, one of `control` (ready map) / `source` (image to extract from) / `pose` (keypoints JSON); optional `references[]`, `strength`, `invert`, inpaint `image` + `mask`/`selection` |
| POST | `/api/controlnet/detect` | `image`, `control_type`, `options` (JSON). Returns the control map, plus editable keypoints in `info.pose` for pose |
| POST | `/api/controlnet/pose/render` | JSON `{pose, width, height}`. Returns the skeleton map as a PNG |
| GET | `/api/controlnet/types` | control types and a template pose |
| POST | `/api/controlnet/download` | fetch the ControlNet weights (7.55 GB) |
| POST | `/api/face` | `image` (target), `face` (reference) files, optional `mask`/`selection`, `prompt`, `keep_hair`, `keep_expression`, `keep_outside` |
| POST | `/api/combine` | `images[]` (2–10), `prompt` |
| POST | `/api/isolate` | `image`, `mask` file or `selection` JSON, `description`, `method`, `crop`, `clip`, `keep_canvas` |
| POST | `/api/background` | `image` (subject), `background` files, optional `mask`/`selection`, `prompt`, `method` (`model`/`composite`), `keep_subject` |
| POST | `/api/upscale` | `image`, one of `scale` / `width` / `height` (width and height together fit the image inside both), `model`, `enhance`, `prompt`, `tile` |
| GET | `/api/upscale/models` | super-resolution models and their download state |
| POST | `/api/upscale/download` | `model`: fetch a super-resolution model ahead of time |
| GET | `/api/models` | variants with download/loaded state |
| POST | `/api/models/load` · `/api/models/download` · `/api/models/unload` | model management |
| GET / DELETE | `/api/jobs/{id}` | poll or cancel a job |
| GET | `/api/jobs/{id}/images/{n}` | result PNG |
| GET | `/api/jobs/{id}/preview` | latest live preview PNG while the job runs |

All operation endpoints take multipart form fields. They also accept the shared options `variant`, `width`, `height`, `resolution`, `steps`, `seed`, `cfg`, `negative_prompt`, `num_images`, `transparent`, `preview`, `preview_every` and `wait`. Each call returns a job (`202`), which you poll; send `wait=true` to block until the job finishes instead. Jobs share a single GPU worker queue.

A `selection` value is either `{"polygons": [[[x, y], ...], ...], "boxes": [[x0, y0, x1, y1], ...], "points": [[x, y], ...]}` or a bare list of polygons. Each entry in `points` auto-selects the object under it. A `mask` image counts white or opaque pixels as selected.

```bash
curl -F prompt="a red fox in snow" -F wait=true http://127.0.0.1:7860/api/generate
curl -F image=@scene.png -F 'selection={"polygons":[[[120,80],[300,90],[290,400],[110,380]]]}' \
     -F description="the knight's sword" -F wait=true http://127.0.0.1:7860/api/isolate
```

## CLI

```bash
qwen-image generate -p "a red fox in snow" -o fox.png [-r ref.png ...] [--transparent]
qwen-image edit -i fox.png -p "make it night time" [--box 100,100,400,400 | --polygon "x,y x,y x,y" | --mask m.png]
qwen-image controlnet -p "a knight in silver armour" --type pose --source dancer.png [-r knight.png] -o knight_dancing.png
qwen-image detect -i dancer.png --type pose -o pose.png --pose-json pose.json   # edit pose.json, then: controlnet --pose-json pose.json
qwen-image face -i scene.png --face hero_portrait.png [--box 300,80,420,220] [--transfer-hair] -o scene_hero.png
qwen-image combine -i cat.png -i hat.png -p "the cat from the first image wearing the hat from the second" -o out.png
qwen-image isolate -i cartoon.png --polygon "10,10 200,15 190,300 20,280" -d "the knight" -o knight.png
qwen-image background -i portrait.png --background beach.png [--method composite] -o portrait_beach.png
qwen-image segment -i photo.png --point 420,310 [--negative 400,200] [--cutout] -o mask.png
qwen-image isolate -i photo.png --point 420,310 -o cutout.png     # --point works wherever a selection does
qwen-image upscale -i photo.png --scale 2.5 [--model nomos-webphoto] [--enhance] -o photo_big.png   # or --width 3840
```

By default each command loads the model in-process. Add `--api http://127.0.0.1:7860` to run the command on a server that already has the model loaded. Use `--json` for machine-readable output, and `qwen-image <command> -h` to list every option.

## Live previews

Each operation can stream previews while it samples. The web UI shows them in a live card in the results grid. The API exposes them as `preview` and `preview_step` on the job, served from `GET /api/jobs/{id}/preview`.

- `preview=fast` (the API and UI default) projects the latents to colours with a linear map. The map is calibrated once per loaded model by VAE-encoding a test image, so each preview costs almost nothing. Previews come every step, at low resolution with approximate colours.
- `preview=full` runs the real VAE decode every `preview_every` steps (5 by default). The output is exact but slower: with CPU offload, the denoiser is swapped out of VRAM for every preview.
- `preview=off` turns previews off.

Both modes show the model's current estimate of the *finished* image, recovered from consecutive steps (`x0 = x - sigma * v`), rather than the noisy latents. That makes the composition readable after only a few steps.

From the CLI, add `--preview-dir steps/` to save every preview as `step_NN.png`. This works locally and with `--api`.

## ControlNet

The ControlNet tab uses [alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union), a 7.55 GB add-on that downloads on first use. It works with every model variant.

- **Control types:** pose (OpenPose/DWPose-style skeleton), canny, depth (Depth Anything V2), line art, soft edges (HED), straight lines (MLSD), scribble and grayscale. The preprocessors come from `controlnet_aux` and download their weights (about 700 MB in total) on first use.
- **Pose editor:** extract a skeleton from a photo, or add a figure by hand. Drag joints to adjust them, Shift-drag to move a whole figure, and double-click to hide a joint. The map is rendered on the server from the keypoints, in the exact style the ControlNet was trained on.
- **Subject references:** to keep a specific character's look, for example to re-pose them, add the character as a subject reference. The prompt is then framed as "a new image of the same subject as in `<image1>`", and strength defaults to 0.8 instead of 1.0 so the reference can still shape the result. Combining references with control is not something the ControlNet was trained for, but in testing it kept identity and clothing well.
- **Inpainting:** repaints a masked region, and can be combined with a control map so the repainted part follows it too.
- **How it works:** the control branch is a port of VideoX-Fun's VACE-style ControlNet onto the stock diffusers transformer (`qwen_image/controlnet.py`). Sixteen parallel control blocks feed skips into every second base block. On GPUs under 32 GB its weights are stored as per-row-scaled FP8 (3.8 GB instead of 7.55 GB), with no visible quality difference in A/B tests. Set `QWEN_IMAGE_CONTROLNET_STORAGE=bf16` to override.

## Auto selection

The **Auto** selection tool selects whatever you click: click a red square on a background and the square is selected. Qwen-Image-2.1 is a generator and cannot segment from a click, so this uses [SAM 2.1](https://huggingface.co/facebook/sam2.1-hiera-large) (Meta, Apache-2.0, 0.9 GB, downloads on first use). SAM picks regions by what they are, not only by colour. A click on a shirt selects the shirt, even if it is patterned.

- **Click** selects the object under the cursor and adds it to the selection. With **Subtract** ticked, the click removes that object from the selection instead.
- **Shift-click** adds another point to the same object, for example to grow a selection from a person's shirt to the whole person. **Alt-click** marks a point as not part of the object, which cuts that part away.
- **By colour (magic wand)** is a model-free alternative that floods the connected region of similar colours.

The image is encoded once per image (about 0.15 s on a GPU, 2 s on a CPU). After that, every click costs about 0.05 s. The model runs on the GPU when there is at least 3 GB of VRAM free, and on the CPU otherwise, so it stays usable while a generation runs. It also moves off the GPU before each generation. Set `QWEN_IMAGE_SEGMENT_DEVICE=cpu|cuda` to fix the device, or `QWEN_IMAGE_SEGMENT_MODEL` to use another SAM 2 checkpoint, such as `facebook/sam2.1-hiera-base-plus`.

## Upscale

Qwen-Image-2.1 generates at most about 2048 px, so it cannot enlarge an image 2.5× or 10× on its own. The Upscale tab uses dedicated super-resolution models through [spandrel](https://github.com/chaiNNer-org/spandrel). Each model is a few tens of MB and downloads on first use.

| Model | Best for | License |
|---|---|---|
| `realesrgan-x4plus` (default) | Photos, renders and illustrations; removes blur, noise and JPEG artefacts | BSD-3-Clause |
| `nomos-webphoto` | Real-world photos, with natural textures (4× RealPLKSR) | CC-BY-4.0 |
| `realesrgan-x4plus-anime` | Anime, cartoons and flat-coloured art | BSD-3-Clause |
| `realesrgan-x2` | Small enlargements of clean images; changes the image less | BSD-3-Clause |

- **Any size:** the models enlarge by a fixed 4× or 2×, so other factors take several passes. The first pass always runs on the original image. Each later pass first shrinks the intermediate so that the model lands on the target (10× is 4×, then 2.5×). A final Lanczos resize gives the exact pixel size. Width and height together fit the image inside both, and the aspect ratio is always kept.
- **Tiles:** the image is processed in overlapping tiles (512 px by default), so input size is limited only by RAM. Lower `tile` if the GPU runs out of memory. Transparency is kept: alpha is resized with Lanczos.
- **Enhance:** Qwen first repaints the image at the chosen resolution (capped at the target size). This restores real texture and removes blur and artefacts, then the upscaler takes it the rest of the way. It is generative, so small details can move or change. An optional prompt describing the image guides it.

## How selections reach the model

**Face** sends the face reference and the target as two condition images, with the target last so the framing is kept. The instruction lists the identity traits to transfer (facial structure, eyes, nose, mouth, jawline, skin tone and marks, plus hair and expression when chosen) and the traits to keep. If the target has a selection, it is outlined in red like a local edit and blended back so nothing outside it changes.

The diffusers pipeline has no mask input. A selection is passed to the model in one of these ways:

- **Edit, `mask_mode=annotate`** (the default): the selected region is outlined in red on the input image, because Qwen-Image-2.1 was trained to follow circled or painted annotations. With `mask_mode=reference`, the mask is passed as a second condition image instead. With `keep_outside`, the result is blended back into the original using a feathered mask, so everything outside the selection keeps its original pixels.
- **Isolate**: the image is first cropped around the selection, so small items fill the model's view. The selection is then outlined in red, and the model is asked for a transparent RGBA extraction using Qwen's recommended RGBA prompt. Its alpha matte is clipped to a slightly dilated copy of the selection. By default the original colours are restored, so the model only decides which pixels are part of the subject. `method=mask` skips the model and uses the selection as the matte.

**Background** has two methods:

- `model` sends the background reference and the subject image as two condition images. The model repaints everything except the subject, matching the lighting and perspective. A selection outlines which subject to keep. `keep_subject` pastes the original pixels inside the selection back, so use it only with a tight selection.
- `composite` cuts the subject out with the same AI matte as **isolate**. Without a selection, the model finds the main subject itself. The cutout is then laid over the background image, scaled to cover the frame. Nothing is regenerated, so the lighting is not adapted.

## Development

```bash
pytest           # runs against a mock pipeline, so no download or GPU is needed
QWEN_IMAGE_ENABLE_MOCK=1 qwen-image serve --variant mock   # UI/API without weights
```
