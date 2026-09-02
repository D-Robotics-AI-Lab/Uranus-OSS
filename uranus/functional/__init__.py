"""
Uranus Inference — Functional Layer

Pure functions for inference orchestration. No internal mutable state.
All runtime state (KV cache, history latents, etc.) is managed by the caller.
"""

from .text_encoder import encode_prompt
from .plucker import (
    compute_plucker_embeddings,
    compute_multi_camera_plucker_embeddings,
    pack_plucker_frames_to_latent_embeddings,
    build_reference_plucker_embeddings,
    build_step_plucker_embeddings,
)
from .skeleton import (
    init_skeleton_cache,
    build_skeleton_context_prefill,
    build_skeleton_context_step,
)
from .lightweight_skeleton import (
    build_lightweight_skeleton_reference_context,
    build_lightweight_skeleton_step_context,
)
from .condition_fusion import fuse_conditions
from .vae_decoder import decode_latents
from ..modules.kv_cache import KVCache
from .stream_state import UranusStreamState
from .dit_inference import (
    dit_prefill,
    dit_decode_step,
    dit_kv_cache_fill,
)
from .stream_runner import (
    UranusStreamConfig,
    generate_frame,
    fill_frame_kv_cache,
    decode_frame,
)

__all__ = [
    "encode_prompt",
    "compute_plucker_embeddings",
    "compute_multi_camera_plucker_embeddings",
    "pack_plucker_frames_to_latent_embeddings",
    "build_reference_plucker_embeddings",
    "build_step_plucker_embeddings",
    "init_skeleton_cache",
    "build_skeleton_context_prefill",
    "build_skeleton_context_step",
    "build_lightweight_skeleton_reference_context",
    "build_lightweight_skeleton_step_context",
    "fuse_conditions",
    "decode_latents",
    "KVCache",
    "dit_prefill",
    "dit_decode_step",
    "dit_kv_cache_fill",
    "UranusStreamConfig",
    "UranusStreamState",
    "generate_frame",
    "fill_frame_kv_cache",
    "decode_frame",
]
