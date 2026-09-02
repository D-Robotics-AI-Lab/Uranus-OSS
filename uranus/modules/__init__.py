"""
Uranus Inference Modules

Migrated from the Uranus project and diffsynth, with all diffsynth
dependencies removed. Only torch and einops are required.
Class names are preserved for direct weight loading compatibility.

Note: HuggingfaceTokenizer requires additional dependencies (transformers, ftfy, regex).
"""

from .t5_encoder import WanTextEncoder
from .vae import WanVideoVAE
from .dit import WanModel, SpatialTemporalWanModel, SpatialTemporalDiTBlock
from .kv_cache import KVCache, PagedLayerKVCache
from .plucker_adapter import PluckerAdapter
from .skeleton_embedding import VacePatchEmbedding, LightweightSkeletonPatchEmbedding
from .scheduler import FlowMatchScheduler

# HuggingfaceTokenizer requires the `transformers` library (optional)
try:
    from .t5_encoder import HuggingfaceTokenizer
except ImportError:
    HuggingfaceTokenizer = None

__all__ = [
    "WanTextEncoder",
    "HuggingfaceTokenizer",
    "WanVideoVAE",
    "WanModel",
    "SpatialTemporalWanModel",
    "SpatialTemporalDiTBlock",
    "KVCache",
    "PagedLayerKVCache",
    "PluckerAdapter",
    "VacePatchEmbedding",
    "LightweightSkeletonPatchEmbedding",
]
