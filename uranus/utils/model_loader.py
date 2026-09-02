"""Model loading + DiT hyperparameter inference.

Modules are constructed with class names preserved for direct
``load_state_dict`` compatibility, and DiT hyperparameters are inferred from
the state dict itself (no model-config file). The returned dict keys are
those the runner's model API consumes: dit, vae, text_encoder, tokenizer,
plucker_adapter, scheduler, skeleton_embedding.

The skeleton controller is chosen from ``skeleton_mode`` ("lightweight" ->
LightweightSkeletonPatchEmbedding, anything else -> VacePatchEmbedding).
"""

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


def build_module(cls, state_dict: dict[str, torch.Tensor], *, strict: bool = True, **ctor_kwargs):
    module = cls(**ctor_kwargs).to(torch.device("cpu")).eval()
    module.load_state_dict(state_dict, strict=strict)
    return module


def load_state(path: Path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    if isinstance(state, dict):
        return state
    raise TypeError(f"Unsupported weight format: {type(state)} @ {path}")


def infer_dit_hparams(
    dit_state_dict: dict[str, torch.Tensor],
) -> tuple[int, int, int, int, int, int, tuple[int, ...], int, int]:
    dim = int(dit_state_dict["patch_embedding.weight"].shape[0])
    in_dim = int(dit_state_dict["patch_embedding.weight"].shape[1])
    patch_size = tuple(int(x) for x in dit_state_dict["patch_embedding.weight"].shape[2:5])
    freq_dim = int(dit_state_dict["time_embedding.0.weight"].shape[1])
    text_dim = int(dit_state_dict["text_embedding.0.weight"].shape[1])
    ffn_dim = int(dit_state_dict["blocks.0.ffn.0.weight"].shape[0])
    out_dim = int(
        dit_state_dict["head.head.weight"].shape[0]
        // (patch_size[0] * patch_size[1] * patch_size[2])
    )
    num_layers = max(int(key.split(".")[1]) for key in dit_state_dict if key.startswith("blocks.")) + 1
    num_heads = max(1, dim // 128)
    return dim, in_dim, ffn_dim, out_dim, text_dim, freq_dim, patch_size, num_heads, num_layers


def load_models(
    weights_dir: str | Path,
    device: torch.device,
    dtype: torch.dtype,
    *,
    skeleton_mode: str = "lightweight",
) -> dict[str, Any]:
    """Load all model modules onto ``device``/``dtype`` from converted weights.

    Expected layout of ``weights_dir``:

        dit.pt  vae.pt  text_encoder.pt  plucker_adapter.pt
        vace_patch_embedding.pt  tokenizer/   (HF tokenizer directory)

    ``vae.pt`` is loaded with ``strict=False`` (VAE state dicts carry extra
    keys across checkpoints); everything else is strict.
    """
    from uranus.modules import (
        FlowMatchScheduler,
        HuggingfaceTokenizer,
        LightweightSkeletonPatchEmbedding,
        PluckerAdapter,
        SpatialTemporalWanModel,
        VacePatchEmbedding,
        WanTextEncoder,
        WanVideoVAE,
    )

    weights_dir = Path(weights_dir).expanduser().resolve()
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Converted weights directory not found: {weights_dir}")
    logger.info("Loading converted weights from %s", weights_dir)

    weights = {
        "dit": load_state(weights_dir / "dit.pt"),
        "vae": load_state(weights_dir / "vae.pt"),
        "text_encoder": load_state(weights_dir / "text_encoder.pt"),
        "plucker_adapter": load_state(weights_dir / "plucker_adapter.pt"),
        "vace_patch_embedding": load_state(weights_dir / "vace_patch_embedding.pt"),
    }
    tokenizer_path = weights_dir / "tokenizer"
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found under converted weights: {tokenizer_path}")

    use_lightweight_skeleton_controller = skeleton_mode == "lightweight"

    dim, in_dim, ffn_dim, out_dim, text_dim, freq_dim, patch_size, num_heads, num_layers = (
        infer_dit_hparams(weights["dit"])
    )
    dit = build_module(
        SpatialTemporalWanModel,
        weights["dit"],
        strict=True,
        dim=dim,
        in_dim=in_dim,
        ffn_dim=ffn_dim,
        out_dim=out_dim,
        text_dim=text_dim,
        freq_dim=freq_dim,
        eps=1e-6,
        patch_size=patch_size,
        num_heads=num_heads,
        num_layers=num_layers,
        has_image_input=False,
    )
    vae = build_module(WanVideoVAE, weights["vae"], strict=False, z_dim=16)
    text_encoder = build_module(WanTextEncoder, weights["text_encoder"], strict=True)
    tokenizer = HuggingfaceTokenizer(name=str(tokenizer_path), seq_len=512, clean="whitespace")
    plucker_adapter = build_module(
        PluckerAdapter, weights["plucker_adapter"], strict=True, in_dim=24, out_dim=dim
    )
    if use_lightweight_skeleton_controller:
        skeleton_embedding = build_module(
            LightweightSkeletonPatchEmbedding,
            weights["vace_patch_embedding"],
            strict=True,
            in_dim=768,
            dim=dim,
        )
    else:
        skeleton_embedding = build_module(
            VacePatchEmbedding,
            weights["vace_patch_embedding"],
            strict=True,
            in_dim=2 * vae.z_dim + 64,
            dim=dim,
        )

    scheduler = FlowMatchScheduler()

    logger.info("Converted weights loaded successfully")
    return {
        "dit": dit.to(device=device, dtype=dtype).eval(),
        "vae": vae.to(device=device, dtype=dtype).eval(),
        "text_encoder": text_encoder.to(device=device, dtype=dtype).eval(),
        "tokenizer": tokenizer,
        "plucker_adapter": plucker_adapter.to(device=device, dtype=dtype).eval(),
        "scheduler": scheduler,
        "skeleton_embedding": skeleton_embedding.to(device=device, dtype=dtype).eval(),
    }
