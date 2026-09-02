import math
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class PagedLayerKVCache:
    batch_size: int
    num_heads: int
    head_dim: int
    page_size: int
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    page_indices_per_request: list[list[int]]
    seq_lens: list[int]
    next_free_page: int
    kv_layout: str = "NHD"
    _dense_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = field(default=None, init=False, repr=False)
    _metadata_cache: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = field(default=None, init=False, repr=False)

    @classmethod
    def empty(
        cls,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        page_size: int,
        device: torch.device,
        dtype: torch.dtype,
        kv_layout: str = "NHD",
    ) -> "PagedLayerKVCache":
        return cls(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size,
            k_cache=torch.empty((0, page_size, num_heads, head_dim), device=device, dtype=dtype),
            v_cache=torch.empty((0, page_size, num_heads, head_dim), device=device, dtype=dtype),
            page_indices_per_request=[[] for _ in range(batch_size)],
            seq_lens=[0 for _ in range(batch_size)],
            next_free_page=0,
            kv_layout=kv_layout,
        )

    @classmethod
    def from_dense(
        cls,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_size: int,
        kv_layout: str = "NHD",
    ) -> "PagedLayerKVCache":
        batch_size, num_heads, max_seq_len, head_dim = k_cache.shape
        page_counts = math.ceil(max_seq_len / page_size) if max_seq_len > 0 else 0
        paged = cls.empty(
            batch_size=batch_size,
            num_heads=num_heads,
            head_dim=head_dim,
            page_size=page_size,
            device=k_cache.device,
            dtype=k_cache.dtype,
            kv_layout=kv_layout,
        )
        if page_counts == 0:
            return paged
        paged._ensure_capacity(batch_size * page_counts)
        for req_idx in range(batch_size):
            req_pages = list(range(req_idx * page_counts, (req_idx + 1) * page_counts))
            paged.page_indices_per_request[req_idx] = req_pages
            paged.seq_lens[req_idx] = max_seq_len
            k_tokens = k_cache[req_idx].permute(1, 0, 2).contiguous()
            v_tokens = v_cache[req_idx].permute(1, 0, 2).contiguous()
            k_pages = torch.zeros(
                (page_counts, page_size, num_heads, head_dim),
                dtype=k_cache.dtype,
                device=k_cache.device,
            )
            v_pages = torch.zeros_like(k_pages)
            k_pages.view(-1, num_heads, head_dim)[:max_seq_len] = k_tokens
            v_pages.view(-1, num_heads, head_dim)[:max_seq_len] = v_tokens
            paged.k_cache[req_pages] = k_pages
            paged.v_cache[req_pages] = v_pages
        paged.next_free_page = batch_size * page_counts
        return paged

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "PagedLayerKVCache":
        return cls(
            batch_size=int(state_dict["batch_size"]),
            num_heads=int(state_dict["num_heads"]),
            head_dim=int(state_dict["head_dim"]),
            page_size=int(state_dict["page_size"]),
            k_cache=state_dict["k_cache"],
            v_cache=state_dict["v_cache"],
            page_indices_per_request=[list(map(int, pages)) for pages in state_dict["page_indices_per_request"]],
            seq_lens=list(map(int, state_dict["seq_lens"])),
            next_free_page=int(state_dict["next_free_page"]),
            kv_layout=state_dict.get("kv_layout", "NHD"),
        )

    @property
    def device(self) -> torch.device:
        return self.k_cache.device

    @property
    def dtype(self) -> torch.dtype:
        return self.k_cache.dtype

    def _invalidate_cached_views(self):
        self._dense_cache = None
        self._metadata_cache = None

    def _ensure_capacity(self, required_pages: int):
        current_pages = self.k_cache.shape[0]
        if required_pages <= current_pages:
            return
        new_total_pages = max(required_pages, current_pages * 2, 1)
        new_k = torch.empty(
            (new_total_pages, self.page_size, self.num_heads, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        new_v = torch.empty_like(new_k)
        if current_pages > 0:
            new_k[:current_pages] = self.k_cache
            new_v[:current_pages] = self.v_cache
        self.k_cache = new_k
        self.v_cache = new_v
        self._invalidate_cached_views()

    def reserve_append(self, append_lengths: list[int]) -> torch.Tensor:
        self._invalidate_cached_views()
        required_new_pages = 0
        for req_idx, append_len in enumerate(append_lengths):
            if append_len <= 0:
                continue
            new_seq_len = self.seq_lens[req_idx] + append_len
            needed_pages = math.ceil(new_seq_len / self.page_size)
            current_pages = len(self.page_indices_per_request[req_idx])
            required_new_pages += max(0, needed_pages - current_pages)
        if required_new_pages > 0:
            self._ensure_capacity(self.next_free_page + required_new_pages)
        for req_idx, append_len in enumerate(append_lengths):
            if append_len <= 0:
                continue
            new_seq_len = self.seq_lens[req_idx] + append_len
            needed_pages = math.ceil(new_seq_len / self.page_size)
            current_pages = len(self.page_indices_per_request[req_idx])
            if needed_pages > current_pages:
                new_pages = list(range(self.next_free_page, self.next_free_page + needed_pages - current_pages))
                self.page_indices_per_request[req_idx].extend(new_pages)
                self.next_free_page += needed_pages - current_pages
            self.seq_lens[req_idx] = new_seq_len
        append_indptr = [0]
        for append_len in append_lengths:
            append_indptr.append(append_indptr[-1] + int(append_len))
        return torch.tensor(append_indptr, dtype=torch.int32, device=self.device)

    def paged_kv_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_cache[:self.next_free_page], self.v_cache[:self.next_free_page]

    def metadata_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._metadata_cache is not None:
            return self._metadata_cache
        page_counts = [len(pages) for pages in self.page_indices_per_request]
        kv_indptr = [0]
        for count in page_counts:
            kv_indptr.append(kv_indptr[-1] + count)
        flat_page_indices = [page for pages in self.page_indices_per_request for page in pages]
        kv_last_page_len = [
            ((seq_len - 1) % self.page_size) + 1 if seq_len > 0 else 0
            for seq_len in self.seq_lens
        ]
        self._metadata_cache = (
            torch.tensor(kv_indptr, dtype=torch.int32, device=self.device),
            torch.tensor(flat_page_indices, dtype=torch.int32, device=self.device),
            torch.tensor(kv_last_page_len, dtype=torch.int32, device=self.device),
            torch.tensor(self.seq_lens, dtype=torch.int32, device=self.device),
        )
        return self._metadata_cache

    def plan_signature(self) -> tuple:
        return (
            self.next_free_page,
            tuple(self.seq_lens),
            tuple(tuple(pages) for pages in self.page_indices_per_request),
        )

    def fork(self) -> "PagedLayerKVCache":
        return type(self)(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            page_size=self.page_size,
            k_cache=self.k_cache,
            v_cache=self.v_cache,
            page_indices_per_request=[list(pages) for pages in self.page_indices_per_request],
            seq_lens=list(self.seq_lens),
            next_free_page=self.next_free_page,
            kv_layout=self.kv_layout,
        )

    def to_dense(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dense_cache is not None:
            return self._dense_cache
        max_seq_len = max(self.seq_lens, default=0)
        k_dense = torch.zeros(
            (self.batch_size, self.num_heads, max_seq_len, self.head_dim),
            device=self.device,
            dtype=self.dtype,
        )
        v_dense = torch.zeros_like(k_dense)
        for req_idx, seq_len in enumerate(self.seq_lens):
            if seq_len == 0:
                continue
            page_ids = self.page_indices_per_request[req_idx]
            k_tokens = self.k_cache[page_ids].reshape(-1, self.num_heads, self.head_dim)[:seq_len]
            v_tokens = self.v_cache[page_ids].reshape(-1, self.num_heads, self.head_dim)[:seq_len]
            k_dense[req_idx, :, :seq_len] = k_tokens.permute(1, 0, 2)
            v_dense[req_idx, :, :seq_len] = v_tokens.permute(1, 0, 2)
        self._dense_cache = (k_dense, v_dense)
        return self._dense_cache

    def sliding_window_drop(self, sliding_window_size: int, num_ref: int, tokens_per_frame: int):
        total_frame_in_cache = max(self.seq_lens, default=0) // tokens_per_frame
        if total_frame_in_cache <= num_ref + sliding_window_size:
            return
        ref_tokens = num_ref * tokens_per_frame
        tail_tokens = sliding_window_size * tokens_per_frame
        if (
            ref_tokens % self.page_size == 0
            and tail_tokens % self.page_size == 0
            and all(seq_len % self.page_size == 0 for seq_len in self.seq_lens)
        ):
            ref_pages = ref_tokens // self.page_size
            tail_pages = tail_tokens // self.page_size
            selected_pages_per_request = []
            flat_selected_pages = []
            for pages in self.page_indices_per_request:
                selected = list(pages[:ref_pages])
                if tail_pages > 0:
                    selected.extend(pages[-tail_pages:])
                selected_pages_per_request.append(selected)
                flat_selected_pages.extend(selected)
            if flat_selected_pages:
                page_index = torch.tensor(flat_selected_pages, dtype=torch.long, device=self.device)
                kept_k = self.k_cache.index_select(0, page_index)
                kept_v = self.v_cache.index_select(0, page_index)
                kept_pages = kept_k.shape[0]
                self.k_cache[:kept_pages] = kept_k
                self.v_cache[:kept_pages] = kept_v
                new_page_indices_per_request = []
                offset = 0
                for selected in selected_pages_per_request:
                    page_count = len(selected)
                    new_page_indices_per_request.append(list(range(offset, offset + page_count)))
                    offset += page_count
                self.page_indices_per_request = new_page_indices_per_request
                kept_seq_len = ref_tokens + tail_tokens
                self.seq_lens = [kept_seq_len for _ in self.seq_lens]
                self.next_free_page = kept_pages
                self._invalidate_cached_views()
                return
        k_dense, v_dense = self.to_dense()
        k_dense = torch.cat([k_dense[:, :, :ref_tokens], k_dense[:, :, -tail_tokens:]], dim=2)
        v_dense = torch.cat([v_dense[:, :, :ref_tokens], v_dense[:, :, -tail_tokens:]], dim=2)
        rebuilt = type(self).from_dense(k_dense, v_dense, page_size=self.page_size, kv_layout=self.kv_layout)
        self.k_cache = rebuilt.k_cache
        self.v_cache = rebuilt.v_cache
        self.page_indices_per_request = rebuilt.page_indices_per_request
        self.seq_lens = rebuilt.seq_lens
        self.next_free_page = rebuilt.next_free_page
        self._invalidate_cached_views()

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "PagedLayerKVCache":
        self.k_cache = KVCache._move_tensor(
            self.k_cache, device=device, dtype=dtype, non_blocking=non_blocking
        )
        self.v_cache = KVCache._move_tensor(
            self.v_cache, device=device, dtype=dtype, non_blocking=non_blocking
        )
        self._invalidate_cached_views()
        return self

    def state_dict(self) -> dict:
        return {
            "format": "paged",
            "batch_size": self.batch_size,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "kv_layout": self.kv_layout,
            "k_cache": self.k_cache[:self.next_free_page],
            "v_cache": self.v_cache[:self.next_free_page],
            "page_indices_per_request": [list(pages) for pages in self.page_indices_per_request],
            "seq_lens": list(self.seq_lens),
            "next_free_page": self.next_free_page,
        }


class KVCache:
    def __init__(self):
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor] | PagedLayerKVCache] = {}

    @staticmethod
    def _move_tensor(
        tensor: torch.Tensor,
        device=None,
        dtype=None,
        non_blocking: bool = False,
    ) -> torch.Tensor:
        if (
            device is not None
            and torch.device(device).type == "cpu"
            and non_blocking
            and tensor.is_cuda
        ):
            # Async D2H into pinned memory (see
            # ``UranusStreamState._move_tensor``).
            pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
            pinned.copy_(tensor, non_blocking=True)
            if dtype is not None and dtype != pinned.dtype:
                pinned = pinned.to(dtype=dtype)
            return pinned
        if dtype is not None and (tensor.is_floating_point() or tensor.is_complex()):
            return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return tensor.to(device=device, non_blocking=non_blocking)

    def get(self, layer_idx: int) -> Optional[tuple[torch.Tensor, torch.Tensor] | PagedLayerKVCache]:
        return self._cache.get(layer_idx)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor] | PagedLayerKVCache:
        return self._cache[layer_idx]

    def __setitem__(self, layer_idx: int, kv: tuple[torch.Tensor, torch.Tensor] | PagedLayerKVCache):
        self._cache[layer_idx] = kv

    def __contains__(self, layer_idx: int) -> bool:
        return layer_idx in self._cache

    def __iter__(self):
        return iter(self._cache)

    def items(self):
        return self._cache.items()

    def state_dict(self) -> dict:
        result = {}
        for layer_idx, layer_cache in self._cache.items():
            if isinstance(layer_cache, PagedLayerKVCache):
                result[layer_idx] = layer_cache.state_dict()
            else:
                k_cache, v_cache = layer_cache
                result[layer_idx] = (k_cache, v_cache)
        return result

    def load_state_dict(self, state_dict: dict):
        loaded = {}
        for layer_idx, layer_cache in state_dict.items():
            if isinstance(layer_cache, dict) and layer_cache.get("format") == "paged":
                loaded[int(layer_idx)] = PagedLayerKVCache.from_state_dict(layer_cache)
            else:
                k_cache, v_cache = layer_cache
                loaded[int(layer_idx)] = (k_cache, v_cache)
        self._cache = loaded
        return self

    @classmethod
    def from_state_dict(cls, state_dict: dict) -> "KVCache":
        return cls().load_state_dict(state_dict)

    def to(self, device=None, dtype=None, non_blocking: bool = False) -> "KVCache":
        if device is None and dtype is None:
            return self
        moved = {}
        for layer_idx, layer_cache in self._cache.items():
            if isinstance(layer_cache, PagedLayerKVCache):
                moved[layer_idx] = layer_cache.to(device=device, dtype=dtype, non_blocking=non_blocking)
            else:
                k_cache, v_cache = layer_cache
                moved[layer_idx] = (
                    self._move_tensor(k_cache, device=device, dtype=dtype, non_blocking=non_blocking),
                    self._move_tensor(v_cache, device=device, dtype=dtype, non_blocking=non_blocking),
                )
        self._cache = moved
        return self

    def sliding_window_drop(self, sliding_window_size: int, num_ref: int, tokens_per_frame: int):
        for layer_id in list(self._cache.keys()):
            layer_cache = self._cache[layer_id]
            if isinstance(layer_cache, PagedLayerKVCache):
                layer_cache.sliding_window_drop(sliding_window_size, num_ref, tokens_per_frame)
                continue
            k_cache, v_cache = layer_cache
            total_frame_in_cache = k_cache.shape[2] // tokens_per_frame
            if total_frame_in_cache <= num_ref + sliding_window_size:
                continue
            k_cache = torch.cat(
                [
                    k_cache[:, :, :num_ref * tokens_per_frame],
                    k_cache[:, :, -sliding_window_size * tokens_per_frame:],
                ],
                dim=2,
            )
            v_cache = torch.cat(
                [
                    v_cache[:, :, :num_ref * tokens_per_frame],
                    v_cache[:, :, -sliding_window_size * tokens_per_frame:],
                ],
                dim=2,
            )
            self._cache[layer_id] = (k_cache, v_cache)
