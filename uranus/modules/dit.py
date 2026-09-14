"""
DiT (Diffusion Transformer) for WanVideo. Includes WanModel (base DiT) and SpatialTemporalWanModel (with spatial/temporal attention factorization). Migrated from diffsynth.models.wan_video_dit and uranus.models.wan. Class names and parameter names are preserved for direct weight loading compatibility.
"""

import torch
import torch.nn as nn
import math
from typing import Tuple, Optional
from einops import rearrange
from torch.nn.attention.flex_attention import flex_attention


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

def rope_apply_fp(x, freqs_fp, num_heads, out_dtype: Optional[torch.dtype] = None):
    if out_dtype is None:
        out_dtype = x.dtype
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    b, s, n, c = x.shape
    # precompute multipliers
    x = x.to(freqs_fp.dtype).reshape(
        b, s, n, -1, 2)
    freq = freqs_fp.view(1, s, 1, -1, 2)
    # apply rotary embedding
    x0 = x[..., 0]
    x1 = x[..., 1]
    f0 = freq[..., 0]
    f1 = freq[..., 1]
    res0 = x0 * f0 - x1 * f1
    res1 = x0 * f1 + x1 * f0
    x = torch.stack([res0.to(out_dtype), res1.to(out_dtype)], dim=-1)
    return x.reshape(b, s, n*c)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class FlexAttentionModule(nn.Module):
    """
    Uranus AttentionModule backed by torch.compile(flex_attention). Supports block_mask and KV cache.
    """
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self._compiled_fn = torch.compile(flex_attention, dynamic=True, mode="default")

    def attention(self, q, k, v, num_heads, block_mask=None, past_kv=None):
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)

        if past_kv is not None:
            past_k, past_v = past_kv
            if past_k is not None and past_v is not None:
                k = torch.cat([past_k, k], dim=2)
                v = torch.cat([past_v, v], dim=2)


        x = self._compiled_fn(q, k, v, block_mask=block_mask)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
        return x, (k, v)

    def forward(self, q, k, v, block_mask=None, past_kv=None, num_heads=None, **kwargs):
        if num_heads is None:
            num_heads = self.num_heads
        x, kv = self.attention(q=q, k=k, v=v, num_heads=num_heads, block_mask=block_mask, past_kv=past_kv)
        return x, kv


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = FlexAttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x, _ = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y, _ = self.attn(q, k_img, v_img)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, gate, residual):
        output = (x + gate * residual).clone()
        return output


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    """
    WanVideo Diffusion Transformer. Two scales: 1.3B (dim=1536, ffn_dim=8960, num_heads=12, num_layers=30) and 14B (dim=5120, ffn_dim=13824, num_heads=40, num_layers=40).
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        # SimpleAdapter removed (not needed for inference)
        self.control_adapter = None

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        f, h, w = x.shape[2], x.shape[3], x.shape[4]
        # Rearrange spatial-temporal dims into token sequence for transformer blocks
        x = rearrange(x, 'b c f h w -> b (f h w) c')
        return x, (f, h, w)

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                **kwargs,
                ):
        t = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep).to(x.dtype))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)

        if self.has_image_input:
            x = torch.cat([x, y], dim=1)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        x, (f, h, w) = self.patchify(x)

        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device).view(torch.float64).to(torch.float32)

        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x


def make_spatial_temporal_block_pattern(num_layers: int, use_spatial_temporal_attention: bool) -> list:
    """Per-block spatial/temporal mode assignment. Depends only on the block index, so
    each block can fix its mode at construction time and the forward pass needs no
    runtime branching."""
    if not use_spatial_temporal_attention:
        return [None] * num_layers
    return ["temporal" if i % 2 == 0 else "spatial" for i in range(num_layers)]


class _FactorizedSelfAttention(nn.Module):
    """Shared QKV projections, RoPE, output projection and FlexAttention backend for the
    spatial/temporal factorized self-attention variants. Submodule names (q/k/v/o/norm_q/
    norm_k/attn) are preserved for direct state-dict compatibility with converted weights.
    """

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = FlexAttentionModule(self.num_heads)


class SpatialSelfAttention(_FactorizedSelfAttention):
    """Intra-frame attention: all cameras attend within the same frame. Does not touch
    the KV cache (no past_kv read, no write-back) and does not record attention."""

    def forward(self, x, freqs, *, kv_cache=None, kv_cache_filling=False,
                layer_idx=None, **axes_lengths):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        q = rope_apply_fp(q, freqs, self.num_heads)
        k = rope_apply_fp(k, freqs, self.num_heads)

        q = rearrange(q, "B (N_CAM F H W) C -> (B F) (N_CAM H W) C", **axes_lengths)
        k = rearrange(k, "B (N_CAM F H W) C -> (B F) (N_CAM H W) C", **axes_lengths)
        v = rearrange(v, "B (N_CAM F H W) C -> (B F) (N_CAM H W) C", **axes_lengths)

        x, _ = self.attn(q, k, v, num_heads=self.num_heads)
        x = rearrange(x, "(B F) (N_CAM H W) C -> B (N_CAM F H W) C", **axes_lengths)

        return self.o(x)


class TemporalSelfAttention(_FactorizedSelfAttention):
    """Inter-frame attention: each camera attends along time. Reads past KV from the
    cache and writes back when filling."""

    @torch.compiler.disable
    def _temporal_attention(self, q, k, v, kv_cache, kv_cache_filling=False, layer_idx=None, **axes_lengths):
        if kv_cache is not None:
            past_kv = kv_cache.get(layer_idx)

        q = rearrange(q, "B (N_CAM F H W) C -> (B N_CAM) (F H W) C", **axes_lengths)
        k = rearrange(k, "B (N_CAM F H W) C -> (B N_CAM) (F H W) C", **axes_lengths)
        v = rearrange(v, "B (N_CAM F H W) C -> (B N_CAM) (F H W) C", **axes_lengths)

        x, cache_out = self.attn(
            q,
            k,
            v,
            past_kv=past_kv,
            num_heads=self.num_heads,
            update_kv=kv_cache_filling,
        )
        x = rearrange(x, "(B N_CAM) (F H W) C -> B (N_CAM F H W) C", **axes_lengths)
        return x, cache_out

    def forward(self, x, freqs, *, kv_cache=None, kv_cache_filling=False,
                layer_idx=None, **axes_lengths):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        q = rope_apply_fp(q, freqs, self.num_heads)
        k = rope_apply_fp(k, freqs, self.num_heads)

        x, cache_out = self._temporal_attention(q, k, v, kv_cache, kv_cache_filling, layer_idx, **axes_lengths)

        output = self.o(x)
        if kv_cache is not None and kv_cache_filling and cache_out is not None:
            kv_cache[layer_idx] = cache_out

        return output


class SpatialTemporalDiTBlock(nn.Module):
    """
    DiT block with factorized SpatialSelfAttention / TemporalSelfAttention. The mode is
    fixed at construction so the forward pass has no runtime branching.
    """

    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int,
                 eps: float = 1e-6, spatial_temporal_mode: str = "temporal"):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        attn_cls = SpatialSelfAttention if spatial_temporal_mode == "spatial" else TemporalSelfAttention
        self.self_attn = attn_cls(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()
        self.head_dim = dim // num_heads
        self.mode = spatial_temporal_mode

    def forward(self, x, context, t_mod, freqs, *,
                kv_cache=None, kv_cache_filling=False,
                layer_idx=None, **axes_lengths):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)

        attn_out = self.self_attn(
            input_x,
            freqs,
            kv_cache=kv_cache,
            kv_cache_filling=kv_cache_filling,
            layer_idx=layer_idx,
            **axes_lengths,
        )

        x = self.gate(x, gate_msa, attn_out)
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class SpatialTemporalWanModel(WanModel):
    """
    WanVideo DiT with spatial/temporal attention factorization. Uses SpatialTemporalDiTBlock. Two scales: 1.3B and 14B.
    """

    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        use_spatial_temporal_attention: bool = True,
    ):
        super().__init__(
            dim, in_dim, ffn_dim, out_dim, text_dim, freq_dim,
            eps, patch_size, num_heads, num_layers, has_image_input,
            has_image_pos_emb, has_ref_conv, add_control_adapter,
            in_dim_control_adapter, seperated_timestep,
            require_vae_embedding, require_clip_embedding,
            fuse_vae_embedding_in_latents,
        )
        spatial_temporal_block_pattern = make_spatial_temporal_block_pattern(
            num_layers, use_spatial_temporal_attention
        )
        self.blocks = nn.ModuleList([
            SpatialTemporalDiTBlock(
                has_image_input, dim, num_heads, ffn_dim, eps,
                spatial_temporal_mode=spatial_temporal_block_pattern[i],
            )
            for i in range(num_layers)
        ])
