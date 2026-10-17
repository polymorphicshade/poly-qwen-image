"""Registry of selectable Qwen-Image-2.1 model variants.

Every variant resolves to the same diffusers ``QwenImage21Pipeline``; they differ in which
checkpoint the denoiser comes from and how (or whether) the large components are quantized.
All weights are fetched from the Hugging Face Hub on first use and cached in the standard
HF cache (``HF_HOME``), so selecting a variant is enough to download it.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field

BASE_REPO = "Qwen/Qwen-Image-2.1"
TURBO_REPO = "Qwen/Qwen-Image-2.1-Turbo"
GGUF_REPO = "unsloth/Qwen-Image-2.1-GGUF"


@dataclass(frozen=True)
class Variant:
    name: str
    label: str
    description: str
    repo: str
    # "none": plain bf16 weights; "bnb4" / "bnb8": bitsandbytes quantization applied at load time;
    # "gguf": the denoiser is loaded from a pre-quantized GGUF file.
    quant: str = "none"
    gguf_file: str | None = None
    # Quantize the Qwen3-VL text encoder with bitsandbytes 4-bit as well.
    quantize_text_encoder: bool = False
    turbo: bool = False
    approx_download_gb: float = 0.0
    hidden: bool = False
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


def _gguf(quant: str, size_gb: float, note: str = "") -> Variant:
    fname = f"qwen-image-2.1-{quant}.gguf"
    return Variant(
        name=f"gguf-{quant.lower()}",
        label=f"GGUF {quant}",
        description=(
            f"Denoiser from {GGUF_REPO}/{fname} (~{size_gb:g} GB), text encoder in 4-bit. {note}"
        ).strip(),
        repo=BASE_REPO,
        quant="gguf",
        gguf_file=fname,
        quantize_text_encoder=True,
        approx_download_gb=round(size_gb + 18.9, 1),
        tags=("quantized", "gguf"),
    )


VARIANTS: dict[str, Variant] = {
    v.name: v
    for v in [
        Variant(
            name="bf16",
            label="Base BF16",
            description="Full-precision official checkpoint, 40 steps. Needs CPU offload on <32 GB GPUs.",
            repo=BASE_REPO,
            approx_download_gb=33.1,
            tags=("official",),
        ),
        Variant(
            name="turbo-bf16",
            label="Turbo BF16",
            description="Official 8-step distilled checkpoint (fixed schedule, no CFG).",
            repo=TURBO_REPO,
            turbo=True,
            approx_download_gb=33.1,
            tags=("official", "turbo"),
        ),
        Variant(
            name="nf4",
            label="Base NF4 (bitsandbytes)",
            description="Official checkpoint quantized to 4-bit NF4 on load (denoiser + text encoder).",
            repo=BASE_REPO,
            quant="bnb4",
            quantize_text_encoder=True,
            approx_download_gb=33.1,
            tags=("quantized", "bitsandbytes"),
        ),
        Variant(
            name="int8",
            label="Base INT8 (bitsandbytes)",
            description="Official checkpoint quantized to 8-bit on load (denoiser), text encoder in 4-bit.",
            repo=BASE_REPO,
            quant="bnb8",
            quantize_text_encoder=True,
            approx_download_gb=33.1,
            tags=("quantized", "bitsandbytes"),
        ),
        Variant(
            name="turbo-nf4",
            label="Turbo NF4 (bitsandbytes)",
            description="8-step Turbo checkpoint quantized to 4-bit NF4 on load. Fastest option on 12-16 GB GPUs.",
            repo=TURBO_REPO,
            quant="bnb4",
            quantize_text_encoder=True,
            turbo=True,
            approx_download_gb=33.1,
            tags=("quantized", "bitsandbytes", "turbo"),
        ),
        _gguf("Q8_0", 7.64, "Near-lossless."),
        _gguf("Q6_K", 6.27),
        _gguf("Q5_K_M", 5.39),
        _gguf("Q4_K_M", 4.2, "Good quality/size balance."),
        _gguf("Q3_K_M", 3.17),
        _gguf("Q2_K", 2.47, "Smallest, noticeably lossy."),
        Variant(
            name="mock",
            label="Mock (no model)",
            description="Procedural stand-in that needs no download. For UI/API/CLI testing.",
            repo="",
            hidden=True,
            tags=("testing",),
        ),
    ]
}

DEFAULT_VARIANT = os.environ.get("QWEN_IMAGE_VARIANT", "gguf-q4_k_m")


def mock_enabled() -> bool:
    return os.environ.get("QWEN_IMAGE_ENABLE_MOCK", "").lower() in ("1", "true", "yes")


def get_variant(name: str | None) -> Variant:
    name = (name or DEFAULT_VARIANT).lower()
    if name not in VARIANTS:
        raise KeyError(f"Unknown variant '{name}'. Choose one of: {', '.join(list_variant_names())}")
    v = VARIANTS[name]
    if v.hidden and not (v.name == "mock" and mock_enabled()):
        raise KeyError(f"Variant '{name}' is not enabled")
    return v


def list_variant_names() -> list[str]:
    return [v.name for v in VARIANTS.values() if not v.hidden or (v.name == "mock" and mock_enabled())]


def list_variants() -> list[Variant]:
    return [VARIANTS[n] for n in list_variant_names()]
