# poly-qwen-image

A custom front end for [Qwen-Image-2.1](https://github.com/QwenLM/Qwen-Image-2.1), built on PyTorch, diffusers and FastAPI. It has four core operations, and each one is available from the Python API, the HTTP API, the web UI and the CLI:

| Operation | What it does |
|---|---|
| **generate** | Text-to-image. You can also pass reference images (up to 10) to borrow a subject, style or layout. Transparent (RGBA) output is supported. |
| **edit** | Edits an image from an instruction. An optional selection limits the edit to one region, and pixels outside that region are kept by default. |
| **pose** | Re-poses the subject of one image to match a pose reference. The reference can be another character, a photo, or a stick-figure or sketch drawing. Identity, clothing, style and background stay the same. |
| **face** | Gives the subject of an image the face from a reference image, to keep characters consistent across images. Hair and expression are kept by default; either can be transferred too. An optional selection picks one face among several. |
| **combine** | Combines subjects from 2–10 images into one new image. |
| **isolate** | Cuts a selected subject out of the image. The model removes the background and everything else and returns an RGBA cutout. This works on sub-elements too, such as an item a character holds or one object in the background. |

You can choose between several model variants, including quantized ones. Each variant downloads from the Hugging Face Hub the first time it is used.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate elsewhere)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e .[dev]
```

`QwenImage21Pipeline` currently ships only on diffusers `main`, so `pyproject.toml` installs diffusers from git.

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

Open http://127.0.0.1:7860. Each **tab** is an independent workspace with all the controls. Use **+** or Ctrl+Alt+T to open a tab, × or middle-click to close one, and double-click a tab to rename it. Tab settings are kept across reloads. In Edit and Isolate, use the lasso, box or brush tools to select a region (L/R/B/E, Ctrl+Z to undo). You can paste images with Ctrl+V. The UI uses only the HTTP API below. Interactive API docs are at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/generate` | `prompt`, optional `references[]` files |
| POST | `/api/edit` | `image`, `prompt`, optional `mask` file or `selection` JSON, `mask_mode`, `keep_outside` |
| POST | `/api/pose` | `image` (subject), `pose` (reference) files, optional `prompt`, `pose_type` (`auto`/`character`/`skeleton`/`sketch`), `keep_background` |
| POST | `/api/face` | `image` (target), `face` (reference) files, optional `mask`/`selection`, `prompt`, `keep_hair`, `keep_expression`, `keep_outside` |
| POST | `/api/combine` | `images[]` (2–10), `prompt` |
| POST | `/api/isolate` | `image`, `mask` file or `selection` JSON, `description`, `method`, `crop`, `clip`, `keep_canvas` |
| GET | `/api/models` | variants with download/loaded state |
| POST | `/api/models/load` · `/api/models/download` · `/api/models/unload` | model management |
| GET / DELETE | `/api/jobs/{id}` | poll or cancel a job |
| GET | `/api/jobs/{id}/images/{n}` | result PNG |
| GET | `/api/jobs/{id}/preview` | latest live preview PNG while the job runs |

All operation endpoints take multipart form fields. They also accept the shared options `variant`, `width`, `height`, `resolution`, `steps`, `seed`, `cfg`, `negative_prompt`, `num_images`, `transparent`, `preview`, `preview_every` and `wait`. Each call returns a job (`202`), which you poll; send `wait=true` to block until the job finishes instead. Jobs share a single GPU worker queue.

A `selection` value is either `{"polygons": [[[x, y], ...], ...], "boxes": [[x0, y0, x1, y1], ...]}` or a bare list of polygons. A `mask` image counts white or opaque pixels as selected.

```bash
curl -F prompt="a red fox in snow" -F wait=true http://127.0.0.1:7860/api/generate
curl -F image=@scene.png -F 'selection={"polygons":[[[120,80],[300,90],[290,400],[110,380]]]}' \
     -F description="the knight's sword" -F wait=true http://127.0.0.1:7860/api/isolate
```

## CLI

```bash
qwen-image generate -p "a red fox in snow" -o fox.png [-r ref.png ...] [--transparent]
qwen-image edit -i fox.png -p "make it night time" [--box 100,100,400,400 | --polygon "x,y x,y x,y" | --mask m.png]
qwen-image pose -i hero.png --pose stick_figure.png --pose-type skeleton -o hero_posed.png
qwen-image face -i scene.png --face hero_portrait.png [--box 300,80,420,220] [--transfer-hair] -o scene_hero.png
qwen-image combine -i cat.png -i hat.png -p "the cat from the first image wearing the hat from the second" -o out.png
qwen-image isolate -i cartoon.png --polygon "10,10 200,15 190,300 20,280" -d "the knight" -o knight.png
```

By default each command loads the model in-process. Add `--api http://127.0.0.1:7860` to run the command on a server that already has the model loaded. Use `--json` for machine-readable output, and `qwen-image <command> -h` to list every option.

## Live previews

Each operation can stream previews while it samples. The web UI shows them in a live card in the results grid. The API exposes them as `preview` and `preview_step` on the job, served from `GET /api/jobs/{id}/preview`.

- `preview=fast` (the API and UI default) projects the latents to colours with a linear map. The map is calibrated once per loaded model by VAE-encoding a test image, so each preview costs almost nothing. Previews come every step, at low resolution with approximate colours.
- `preview=full` runs the real VAE decode every `preview_every` steps (5 by default). The output is exact but slower: with CPU offload, the denoiser is swapped out of VRAM for every preview.
- `preview=off` turns previews off.

Both modes show the model's current estimate of the *finished* image, recovered from consecutive steps (`x0 = x - sigma * v`), rather than the noisy latents. That makes the composition readable after only a few steps.

From the CLI, add `--preview-dir steps/` to save every preview as `step_NN.png`. This works locally and with `--api`.

## How selections reach the model

**Pose** sends the pose reference and the subject image as two condition images, with the subject last so the output keeps its aspect ratio. The instruction asks the model to copy only the pose: the placement and angles of the head, torso, limbs, hands and feet, not the reference's appearance.

**Face** sends the face reference and the target as two condition images, with the target last so the framing is kept. The instruction lists the identity traits to transfer (facial structure, eyes, nose, mouth, jawline, skin tone and marks, plus hair and expression when chosen) and the traits to keep. If the target has a selection, it is outlined in red like a local edit and blended back so nothing outside it changes.

The diffusers pipeline has no mask input. A selection is passed to the model in one of these ways:

- **Edit, `mask_mode=annotate`** (the default): the selected region is outlined in red on the input image, because Qwen-Image-2.1 was trained to follow circled or painted annotations. With `mask_mode=reference`, the mask is passed as a second condition image instead. With `keep_outside`, the result is blended back into the original using a feathered mask, so everything outside the selection keeps its original pixels.
- **Isolate**: the image is first cropped around the selection, so small items fill the model's view. The selection is then outlined in red, and the model is asked for a transparent RGBA extraction using Qwen's recommended RGBA prompt. Its alpha matte is clipped to a slightly dilated copy of the selection. By default the original colours are restored, so the model only decides which pixels are part of the subject. `method=mask` skips the model and uses the selection as the matte.

## Development

```bash
pytest           # runs against a mock pipeline, so no download or GPU is needed
QWEN_IMAGE_ENABLE_MOCK=1 qwen-image serve --variant mock   # UI/API without weights
```
