"""ControlNet-Union for Qwen-Image-2.1 (alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union).

Ported from VideoX-Fun's ``QwenImage21ControlTransformer2DModel`` onto the stock diffusers transformer,
so it works with every variant here (BF16, bitsandbytes, GGUF):

* A parallel chain of 16 control blocks (same structure as the base blocks) reads the control
  conditioning. The first block seeds its stream with the base joint stream through ``before_proj``;
  every control block emits a skip through ``after_proj``.
* The skips are added to the outputs of base blocks 0, 2, 4, ... 30, scaled by ``control_context_scale``.
* The conditioning is 129 channels per target latent token: VAE latents of the control image (64),
  an inpaint keep-mask (1) and VAE latents of the masked inpaint image (64). With no inpaint input the
  mask is "regenerate everything" and the masked latents are zeros; with no control image the
  control latents are zeros (pure inpainting).

The branch is kept in CPU RAM and only attached to the transformer for control runs, so ordinary
generations don't move 4-8 GB of unused weights to the GPU.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)

CONTROLNET_REPO = "alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union"
CONTROLNET_FILE = "Qwen-Image-2.1-Fun-Controlnet-Union.safetensors"
CONTROLNET_GB = 7.55
CONTROL_LAYERS = list(range(0, 32, 2))
CONTROL_IN_DIM = 129
FP8_MAX = 448.0


# ---------------------------------------------------------------------- download


def is_downloaded() -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(CONTROLNET_REPO, CONTROLNET_FILE), str)


def download(report=None) -> str:
    from huggingface_hub import hf_hub_download

    if report and not is_downloaded():
        report(f"Downloading ControlNet ({CONTROLNET_GB:g} GB)")
    return hf_hub_download(CONTROLNET_REPO, CONTROLNET_FILE)


# ---------------------------------------------------------------------- weights


class ScaledFP8Linear(nn.Module):
    """A Linear whose weight is stored as float8 with one scale per output row.

    Halves the memory of a bf16 Linear. Per-row scaling keeps small weights out of float8's
    subnormal range, which plain float8 casting would flush to zero.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, compute_dtype=torch.bfloat16):
        super().__init__()
        w = weight.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / FP8_MAX
        self.register_buffer("weight_fp8", (w / scale).to(torch.float8_e4m3fn))
        self.register_buffer("scale", scale.to(compute_dtype))
        self.bias = nn.Parameter(bias.to(compute_dtype), requires_grad=False) if bias is not None else None
        self.compute_dtype = compute_dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight_fp8.to(self.compute_dtype) * self.scale
        return F.linear(x.to(self.compute_dtype), w, self.bias)


def _resolve_storage(storage: str) -> str:
    storage = (storage or os.environ.get("QWEN_IMAGE_CONTROLNET_STORAGE", "auto")).lower()
    if storage != "auto":
        return storage
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory / 1024**3 >= 32:
        return "bf16"
    return "fp8"


class ControlBranch(nn.Module):
    def __init__(self, dim=4096, heads=32, head_dim=128, mlp_ratio=3, eps=1e-6):
        super().__init__()
        from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21TransformerBlock

        blocks = []
        for n in range(len(CONTROL_LAYERS)):
            block = QwenImage21TransformerBlock(dim, heads, head_dim, mlp_ratio, eps)
            if n == 0:
                block.before_proj = nn.Linear(dim, dim)
            block.after_proj = nn.Linear(dim, dim)
            blocks.append(block)
        self.control_blocks = nn.ModuleList(blocks)
        self.control_img_in = nn.Linear(CONTROL_IN_DIM, dim)
        self.layer_to_hint = {layer: n for n, layer in enumerate(CONTROL_LAYERS)}

    def hints(self, control_joint: torch.Tensor, x_joint: torch.Tensor, block_kwargs: dict) -> list[torch.Tensor]:
        c = self.control_blocks[0].before_proj(control_joint) + x_joint
        skips = []
        for block in self.control_blocks:
            c = block(c, **block_kwargs)
            skips.append(block.after_proj(c))
        return skips


def load_branch(storage: str = "auto", report=None) -> ControlBranch:
    """Build the control branch on the meta device and fill it from the checkpoint."""
    from safetensors.torch import load_file

    path = download(report)
    storage = _resolve_storage(storage)
    if report:
        report(f"Loading ControlNet ({storage})")
    with torch.device("meta"):
        branch = ControlBranch()
    state = load_file(path)
    branch.load_state_dict({k: v.to(torch.bfloat16) for k, v in state.items()}, assign=True, strict=True)
    del state
    if storage == "fp8":
        _to_fp8(branch)
    branch.eval().requires_grad_(False)
    return branch


def _to_fp8(module: nn.Module) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and child.in_features >= 1024:
            setattr(module, name, ScaledFP8Linear(child.weight.data, child.bias.data if child.bias is not None else None))
        else:
            _to_fp8(child)


# ---------------------------------------------------------------------- transformer forward


def controlled_forward(
    self,
    hidden_states,
    encoder_hidden_states,
    timestep,
    img_shapes,
    img_mask,
    encoder_hidden_states_mask=None,
    attention_kwargs=None,
    kv_cache=None,
    kv_cache_mode=None,
    return_dict=True,
):
    """``QwenImage21Transformer2DModel.forward`` plus control skips (prefill path, no KV cache).

    Reads the attached branch and conditioning from ``self.qi_control_branch`` / ``self._qi_control``.
    The conditioning is scattered onto the *target* image's tokens only, so reference images given
    as ordinary condition images still work alongside it.
    """
    from diffusers.models.modeling_outputs import Transformer2DModelOutput
    from diffusers.models.transformers.transformer_qwenimage21 import (
        _IMG_TOKENS_PER_SLOT,
        QwenImage21FlexAttnProcessor,
        _qwenimage21_prefix_segments,
        build_qwenimage21_block_causal_mask,
    )

    if kv_cache is not None or kv_cache_mode is not None:
        raise ValueError("ControlNet runs need use_kv_cache=False")
    branch = self.qi_control_branch
    context, scale = self._qi_control
    batch_size = hidden_states.shape[0]

    hidden_states = self.img_in(hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
    image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)
    target_tokens = math.prod(img_shapes[0][-1])
    joint = torch.cat(
        [encoder_hidden_states, encoder_hidden_states.new_zeros(batch_size, target_tokens // 4, encoder_hidden_states.shape[2])],
        dim=1,
    )
    joint = joint.repeat_interleave(repeats, dim=1)
    joint[:, image_pad_mask] = hidden_states

    rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device)
    image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

    ctx = context.to(device=joint.device, dtype=joint.dtype)
    if ctx.shape[0] != batch_size:
        ctx = ctx.expand(batch_size, -1, -1)
    control_joint = torch.zeros_like(joint)
    control_joint[:, target_token_mask] = branch.control_img_in(ctx)

    timestep = timestep.to(hidden_states.dtype)
    if self.config.causal_condition:
        timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
        modulation_mask = target_token_mask
    else:
        modulation_mask = None
    temb = self.time_text_embed(timestep, hidden_states)
    modulation = self.modulation(temb)

    joint_key_valid = None
    if encoder_hidden_states_mask is not None:
        joint_key_valid = torch.ones(batch_size, image_pad_mask.shape[0], dtype=torch.bool, device=hidden_states.device)
        text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
        vlm_text_positions = ~img_mask[0][: encoder_hidden_states_mask.shape[1]]
        joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

    prefix_len = int((~target_token_mask).sum())
    processors = [block.attn.processor for block in self.transformer_blocks]
    needs_block_mask = any(isinstance(p, QwenImage21FlexAttnProcessor) for p in processors)
    attention_mask = (
        build_qwenimage21_block_causal_mask(image_ids, joint_key_valid, batch_size, hidden_states.device)
        if needs_block_mask
        else None
    )
    segments = None if all(isinstance(p, QwenImage21FlexAttnProcessor) for p in processors) else _qwenimage21_prefix_segments(image_ids, prefix_len)
    block_kwargs = dict(
        modulation=modulation,
        rotary_emb=rotary_emb,
        attention_mask=attention_mask,
        target_token_mask=modulation_mask,
        segments=segments,
        key_valid=joint_key_valid,
    )

    hints = branch.hints(control_joint, joint, block_kwargs) if scale else None
    for index, block in enumerate(self.transformer_blocks):
        joint = block(hidden_states=joint, **block_kwargs)
        if hints is not None and index in branch.layer_to_hint:
            joint = joint + hints[branch.layer_to_hint[index]] * scale

    joint = self.norm_out(joint, temb, modulation_mask)
    output = self.proj_out(joint)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


@contextmanager
def attached(transformer, branch: ControlBranch, context: torch.Tensor, scale: float):
    """Temporarily give ``transformer`` the control branch and conditioning.

    The branch is registered as a submodule so CPU-offload hooks move it to the GPU together with
    the transformer. Accelerate's hooks call ``_old_forward``, so that is what gets swapped when present.
    """
    target = "_old_forward" if hasattr(transformer, "_old_forward") else "forward"
    original = getattr(transformer, target)
    # Under CPU offload this is the CPU and the hook moves both later; without offload it is the GPU.
    branch.to(next(transformer.parameters()).device)
    transformer.qi_control_branch = branch
    transformer._qi_control = (context, scale)
    setattr(transformer, target, functools.partial(controlled_forward, transformer))
    try:
        yield
    finally:
        setattr(transformer, target, original)
        del transformer._qi_control
        del transformer.qi_control_branch
        branch.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------- conditioning


def _to_vae_input(pipe, image, width, height, device, dtype, keep=None):
    t = pipe.image_processor.preprocess(image.convert("RGB"), height=height, width=width)  # [1, 3, H, W] in [-1, 1]
    if keep is not None:
        t = t * keep
    t = torch.cat([t, torch.ones_like(t[:, :1])], dim=1)  # opaque alpha: the 2.1 VAE reads RGBA
    return t.unsqueeze(2).to(device=device, dtype=dtype)


@torch.no_grad()
def build_context(pipe, width: int, height: int, control_image=None, inpaint_image=None, mask=None) -> torch.Tensor:
    """Pack the 129-channel conditioning for one sample: ``[1, (h/16)*(w/16), 129]``."""
    device = pipe._execution_device
    dtype = pipe.vae.dtype
    lat_h, lat_w = height // 16, width // 16

    if mask is not None:
        m = np.asarray(mask.convert("L").resize((width, height)), dtype=np.float32) / 255.0
        regen = torch.from_numpy((m >= 0.5).astype(np.float32))[None, None]
    else:
        regen = torch.ones(1, 1, height, width)

    if inpaint_image is not None:
        inpaint = pipe._encode_vae_image(
            _to_vae_input(pipe, inpaint_image, width, height, device, dtype, keep=(regen < 0.5).float()), None
        )
    else:
        inpaint = torch.zeros(1, pipe.latent_channels, 1, lat_h, lat_w, device=device, dtype=dtype)

    if control_image is not None:
        control = pipe._encode_vae_image(_to_vae_input(pipe, control_image, width, height, device, dtype), None)
    else:
        control = torch.zeros_like(inpaint)

    keep_latent = F.interpolate(1 - regen, size=(lat_h, lat_w), mode="nearest").unsqueeze(2).to(device, dtype)
    ctx = torch.cat([control, keep_latent, inpaint], dim=1)  # [1, 129, 1, h, w]
    return ctx.view(1, CONTROL_IN_DIM, lat_h * lat_w).transpose(1, 2).contiguous()
