"""Downloading variants from the Hugging Face Hub and turning them into pipelines."""

from __future__ import annotations

import logging
from collections.abc import Callable

from .variants import BASE_REPO, GGUF_REPO, Variant

log = logging.getLogger(__name__)

# Never needed for inference.
_IGNORE_ALWAYS = ["assets/*", "*.png", "*.jpg", "*.webp", "*.md"]
_TRANSFORMER_WEIGHTS = ["transformer/*.safetensors"]


def _base_ignore(variant: Variant) -> list[str]:
    ignore = list(_IGNORE_ALWAYS)
    if variant.quant == "gguf":
        # The denoiser comes from the GGUF file; only its config is needed from the base repo.
        ignore += _TRANSFORMER_WEIGHTS
    return ignore


def is_downloaded(variant: Variant) -> bool:
    """True when every file the variant needs is already in the local HF cache."""
    if variant.name == "mock":
        return True
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    try:
        snapshot_download(variant.repo, ignore_patterns=_base_ignore(variant), local_files_only=True)
    except Exception:
        return False
    if variant.gguf_file:
        hit = try_to_load_from_cache(GGUF_REPO, variant.gguf_file)
        if not isinstance(hit, str):
            return False
    return True


def download(variant: Variant, report: Callable[[str], None] | None = None) -> tuple[str, str | None]:
    """Fetch (or reuse cached) files for ``variant``; returns (pipeline dir, gguf path or None)."""
    from huggingface_hub import hf_hub_download, snapshot_download

    report = report or (lambda msg: log.info(msg))
    if variant.name == "mock":
        return "", None
    report(f"Downloading {variant.repo} (cached files are reused)")
    path = snapshot_download(variant.repo, ignore_patterns=_base_ignore(variant))
    gguf_path = None
    if variant.gguf_file:
        report(f"Downloading {GGUF_REPO}/{variant.gguf_file}")
        gguf_path = hf_hub_download(GGUF_REPO, variant.gguf_file)
    report("Download complete")
    return path, gguf_path


def resolve_offload(offload: str) -> str:
    """``auto`` keeps everything on the GPU only when it can plausibly hold the full model."""
    if offload != "auto":
        return offload
    import torch

    if not torch.cuda.is_available():
        return "none"
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return "none" if total_gb >= 40 else "model"


def _dequantize_non_linear_gguf_params(model, dtype) -> None:
    """Turn GGUF-typed parameters outside linear layers into plain tensors.

    diffusers keeps BF16 tensors from a GGUF file as raw byte buffers and only GGUFLinear knows
    to dequantize them, so elementwise uses such as norm weights (e.g. ``txt_in.text_norm``)
    would otherwise operate on bytes and fail with a shape mismatch.
    """
    import torch
    from diffusers.quantizers.gguf.utils import GGUFLinear, GGUFParameter, dequantize_gguf_tensor

    for module in model.modules():
        if isinstance(module, GGUFLinear):
            continue
        for name, param in list(module.named_parameters(recurse=False)):
            if isinstance(param, GGUFParameter):
                plain = dequantize_gguf_tensor(param).to(dtype)
                setattr(module, name, torch.nn.Parameter(plain.as_subclass(torch.Tensor), requires_grad=False))


def build_pipeline(variant: Variant, offload: str = "auto", report: Callable[[str], None] | None = None):
    report = report or (lambda msg: log.info(msg))
    if variant.name == "mock":
        from .mock import MockPipeline

        return MockPipeline()

    import torch
    from diffusers import PipelineQuantizationConfig, QwenImage21Pipeline

    path, gguf_path = download(variant, report)
    dtype = torch.bfloat16
    components: dict = {}
    quant_mapping: dict = {}

    if variant.quant == "gguf":
        from diffusers import GGUFQuantizationConfig, QwenImage21Transformer2DModel

        report(f"Loading GGUF denoiser {variant.gguf_file}")
        components["transformer"] = QwenImage21Transformer2DModel.from_single_file(
            gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            config=path,
            subfolder="transformer",
            torch_dtype=dtype,
        )
        _dequantize_non_linear_gguf_params(components["transformer"], dtype)
    elif variant.quant in ("bnb4", "bnb8"):
        from diffusers import BitsAndBytesConfig as DiffusersBnbConfig

        if variant.quant == "bnb4":
            quant_mapping["transformer"] = DiffusersBnbConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
            )
        else:
            quant_mapping["transformer"] = DiffusersBnbConfig(load_in_8bit=True)

    if variant.quantize_text_encoder:
        from transformers import BitsAndBytesConfig as TransformersBnbConfig

        quant_mapping["text_encoder"] = TransformersBnbConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype
        )

    report(f"Loading pipeline from {variant.repo}")
    pipe = QwenImage21Pipeline.from_pretrained(
        path,
        dtype=dtype,
        quantization_config=PipelineQuantizationConfig(quant_mapping=quant_mapping) if quant_mapping else None,
        **components,
    )

    mode = resolve_offload(offload)
    report(f"Placing model (offload={mode})")
    if mode == "model":
        pipe.enable_model_cpu_offload()
    elif mode == "sequential":
        pipe.enable_sequential_cpu_offload()
    elif torch.cuda.is_available():
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


__all__ = ["BASE_REPO", "build_pipeline", "download", "is_downloaded", "resolve_offload"]
