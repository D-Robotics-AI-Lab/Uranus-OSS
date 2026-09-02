import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Iterable

import torch

from ..modules.kv_cache import KVCache


@dataclass
class StreamMeta:
    num_cameras: int = 3
    height: int = 224
    width: int = 224
    skeleton_mode: str = "lightweight"
    num_inference_steps: int = 50
    tokens_per_frame: int = 270
    teacher_forcing_window_size: int = 8
    temporal_interval: int = 4
    spatial_interval: int = 8


@dataclass
class StreamPrefillState:
    reference_latents: torch.Tensor | None = None
    reference_skeleton_latents: torch.Tensor | None = None
    reference_plucker_embedding: torch.Tensor | None = None
    reference_fused_context: torch.Tensor | None = None


@dataclass
class StreamDecodeState:
    dec_feat_map: object | None = None
    dec_feat_idx: list | None = None
    decoded_latent_frames: int = 0


@dataclass
class StreamRuntimeState:
    prompt: str | None = None
    context: torch.Tensor | None = None
    kv_cache: KVCache | None = None
    rope_freqs: tuple[torch.Tensor, ...] | None = None
    current_frame_start_index: int = 0
    num_reference: int = 3
    prefill: StreamPrefillState = field(default_factory=StreamPrefillState)
    decode: StreamDecodeState = field(default_factory=StreamDecodeState)


class UranusStreamState:
    SECTIONS: ClassVar[tuple[str, ...]] = ("meta", "core", "prefill", "decode")

    def __init__(
        self,
        meta: StreamMeta | None = None,
        runtime: StreamRuntimeState | None = None,
        **kwargs,
    ):
        object.__setattr__(self, "meta", meta if meta is not None else StreamMeta())
        object.__setattr__(self, "runtime", runtime if runtime is not None else StreamRuntimeState())
        for key, value in kwargs.items():
            if key not in self._field_names():
                raise TypeError(f"Unexpected UranusStreamState field: {key}")
            setattr(self, key, value)

    @classmethod
    def _meta_field_names(cls) -> set[str]:
        return {field.name for field in fields(StreamMeta)}

    @classmethod
    def _runtime_field_names(cls) -> set[str]:
        return {
            field.name
            for field in fields(StreamRuntimeState)
            if field.name not in {"prefill", "decode"}
        }

    @classmethod
    def _prefill_field_names(cls) -> set[str]:
        return {field.name for field in fields(StreamPrefillState)}

    @classmethod
    def _decode_field_names(cls) -> set[str]:
        return {field.name for field in fields(StreamDecodeState)}

    @classmethod
    def _field_names(cls) -> set[str]:
        return (
            cls._meta_field_names()
            | cls._runtime_field_names()
            | cls._prefill_field_names()
            | cls._decode_field_names()
        )

    def _field_owner(self, name: str):
        if name in self._meta_field_names():
            return self.meta
        if name in self._runtime_field_names():
            return self.runtime
        if name in self._prefill_field_names():
            return self.runtime.prefill
        if name in self._decode_field_names():
            return self.runtime.decode
        return None

    def __getattr__(self, name: str):
        owner = self._field_owner(name)
        if owner is None:
            raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")
        return getattr(owner, name)

    def __setattr__(self, name: str, value):
        if name in {"meta", "runtime"} or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        owner = self._field_owner(name)
        if owner is None:
            object.__setattr__(self, name, value)
            return
        setattr(owner, name, value)

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
            # Async D2H into pinned memory — the only truly asynchronous D2H
            # path. The resulting pinned tensors make the later H2D reload
            # (``SessionRecord.reload_to_device``) asynchronous as well.
            # (``tensor.pin_memory()`` rejects CUDA inputs on torch 2.10, so
            # allocate a pinned target and copy asynchronously instead.)
            pinned = torch.empty_like(tensor, device="cpu", pin_memory=True)
            pinned.copy_(tensor, non_blocking=True)
            if dtype is not None and dtype != pinned.dtype:
                pinned = pinned.to(dtype=dtype)
            return pinned
        if dtype is not None and (tensor.is_floating_point() or tensor.is_complex()):
            return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return tensor.to(device=device, non_blocking=non_blocking)

    @classmethod
    def _move_value(cls, value: Any, device=None, dtype=None, non_blocking: bool = False):
        if isinstance(value, torch.Tensor):
            return cls._move_tensor(value, device=device, dtype=dtype, non_blocking=non_blocking)
        if isinstance(value, KVCache):
            return value.to(device=device, dtype=dtype, non_blocking=non_blocking)
        if is_dataclass(value):
            for meta_field in fields(value):
                setattr(
                    value,
                    meta_field.name,
                    cls._move_value(
                        getattr(value, meta_field.name),
                        device=device,
                        dtype=dtype,
                        non_blocking=non_blocking,
                    ),
                )
            return value
        if isinstance(value, tuple):
            return tuple(
                cls._move_value(item, device=device, dtype=dtype, non_blocking=non_blocking)
                for item in value
            )
        if isinstance(value, list):
            return [
                cls._move_value(item, device=device, dtype=dtype, non_blocking=non_blocking)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: cls._move_value(item, device=device, dtype=dtype, non_blocking=non_blocking)
                for key, item in value.items()
            }
        return value

    @classmethod
    def _normalize_sections(cls, sections: str | Iterable[str] | None) -> set[str]:
        if sections is None:
            return set(cls.SECTIONS)
        if isinstance(sections, str):
            sections = {sections}
        else:
            sections = set(sections)
        if "all" in sections:
            sections = set(cls.SECTIONS)
        if "runtime" in sections:
            sections.remove("runtime")
            sections.update({"core", "prefill", "decode"})
        invalid = sections - set(cls.SECTIONS)
        if invalid:
            raise ValueError(f"Unsupported state sections: {sorted(invalid)}")
        return sections

    @classmethod
    def _serialize_value(cls, value: Any, device=None, dtype=None):
        if isinstance(value, KVCache):
            value = value.state_dict()
        elif is_dataclass(value):
            value = {field.name: cls._serialize_value(getattr(value, field.name), device=device, dtype=dtype) for field in fields(value)}
            return value
        elif isinstance(value, tuple):
            value = tuple(cls._serialize_value(item, device=device, dtype=dtype) for item in value)
        elif isinstance(value, list):
            value = [cls._serialize_value(item, device=device, dtype=dtype) for item in value]
        elif isinstance(value, dict):
            value = {
                key: cls._serialize_value(item, device=device, dtype=dtype)
                for key, item in value.items()
            }
        if device is not None or dtype is not None:
            value = cls._move_value(value, device=device, dtype=dtype)
        return value

    @classmethod
    def _deserialize_runtime_value(cls, field_name: str, value: Any):
        if field_name == "kv_cache" and value is not None and not isinstance(value, KVCache):
            return KVCache.from_state_dict(value)
        return value

    @staticmethod
    def _load_dataclass(target, payload: dict[str, Any], value_parser=None):
        for target_field in fields(target):
            if target_field.name not in payload:
                continue
            value = payload[target_field.name]
            if value_parser is not None:
                value = value_parser(target_field.name, value)
            setattr(target, target_field.name, value)

    def state_dict(
        self,
        device=None,
        dtype=None,
        sections: str | Iterable[str] | None = None,
    ) -> dict[str, Any]:
        selected = self._normalize_sections(sections)
        runtime_payload = {}
        if "core" in selected:
            runtime_payload["core"] = {
                "prompt": self._serialize_value(self.runtime.prompt, device=device, dtype=dtype),
                "context": self._serialize_value(self.runtime.context, device=device, dtype=dtype),
                "kv_cache": self._serialize_value(self.runtime.kv_cache, device=device, dtype=dtype),
                "rope_freqs": self._serialize_value(self.runtime.rope_freqs, device=device, dtype=dtype),
                "current_frame_start_index": self.runtime.current_frame_start_index,
                "num_reference": self.runtime.num_reference,
            }
        if "prefill" in selected:
            runtime_payload["prefill"] = self._serialize_value(self.runtime.prefill, device=device, dtype=dtype)
        if "decode" in selected:
            runtime_payload["decode"] = self._serialize_value(self.runtime.decode, device=device, dtype=dtype)
        payload = {"version": 2}
        if "meta" in selected:
            payload["meta"] = self._serialize_value(self.meta, device=device, dtype=dtype)
        if runtime_payload:
            payload["runtime"] = runtime_payload
        return payload

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
        sections: str | Iterable[str] | None = None,
    ) -> "UranusStreamState":
        payload = state_dict.get("data", state_dict)
        selected = self._normalize_sections(sections)
        if "meta" in payload or "runtime" in payload:
            if "meta" in selected and "meta" in payload:
                self._load_dataclass(self.meta, payload["meta"])
            runtime_payload = payload.get("runtime", {})
            if "core" in selected and "core" in runtime_payload:
                core_payload = runtime_payload["core"]
                for key in ("prompt", "context", "kv_cache", "rope_freqs", "current_frame_start_index", "num_reference"):
                    if key not in core_payload:
                        continue
                    value = self._deserialize_runtime_value(key, core_payload[key])
                    setattr(self.runtime, key, value)
            if "prefill" in selected and "prefill" in runtime_payload:
                self._load_dataclass(self.runtime.prefill, runtime_payload["prefill"])
            if "decode" in selected and "decode" in runtime_payload:
                self._load_dataclass(self.runtime.decode, runtime_payload["decode"])
            return self

        for field_name in self._field_names():
            if field_name not in payload:
                continue
            setattr(self, field_name, self._deserialize_runtime_value(field_name, payload[field_name]))
        return self

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, Any],
        sections: str | Iterable[str] | None = None,
    ) -> "UranusStreamState":
        return cls().load_state_dict(state_dict, sections=sections)

    def save(
        self,
        path: str | os.PathLike,
        move_to_cpu: bool = True,
        sections: str | Iterable[str] | None = None,
    ):
        device = "cpu" if move_to_cpu else None
        torch.save(self.state_dict(device=device, sections=sections), path)
        return path

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        map_location: str | torch.device | None = "cpu",
        device=None,
        dtype=None,
        sections: str | Iterable[str] | None = None,
    ) -> "UranusStreamState":
        state_dict = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(state_dict, cls):
            state = state_dict
        elif isinstance(state_dict, dict):
            state = cls.from_state_dict(state_dict, sections=sections)
        else:
            raise TypeError(f"Unsupported stream state type: {type(state_dict)}")
        if device is not None or dtype is not None:
            state = state.to(device=device, dtype=dtype, sections=sections)
        return state

    def to(
        self,
        device=None,
        dtype=None,
        non_blocking: bool = False,
        sections: str | Iterable[str] | None = None,
    ) -> "UranusStreamState":
        if device is None and dtype is None:
            return self
        selected = self._normalize_sections(sections)
        if "meta" in selected:
            self._move_value(self.meta, device=device, dtype=dtype, non_blocking=non_blocking)
        if "core" in selected:
            for key in ("context", "kv_cache", "rope_freqs"):
                setattr(
                    self.runtime,
                    key,
                    self._move_value(
                        getattr(self.runtime, key),
                        device=device,
                        dtype=dtype,
                        non_blocking=non_blocking,
                    ),
                )
        if "prefill" in selected:
            self._move_value(self.runtime.prefill, device=device, dtype=dtype, non_blocking=non_blocking)
        if "decode" in selected:
            self._move_value(self.runtime.decode, device=device, dtype=dtype, non_blocking=non_blocking)
        return self

    def cpu(self, sections: str | Iterable[str] | None = None) -> "UranusStreamState":
        return self.to(device="cpu", sections=sections)

    def clear_prefill(self) -> "UranusStreamState":
        self.runtime.prefill = StreamPrefillState()
        return self

    def prefill(
        self,
        reference_latents: torch.Tensor | None = None,
        reference_skeleton_latents: torch.Tensor | None = None,
        reference_plucker_embedding: torch.Tensor | None = None,
        reference_fused_context: torch.Tensor | None = None,
    ) -> "UranusStreamState":
        if reference_latents is not None:
            self.reference_latents = reference_latents
        if reference_skeleton_latents is not None:
            self.reference_skeleton_latents = reference_skeleton_latents
        if reference_plucker_embedding is not None:
            self.reference_plucker_embedding = reference_plucker_embedding
        if reference_fused_context is not None:
            self.reference_fused_context = reference_fused_context
        return self

    def clear_decode(self) -> "UranusStreamState":
        self.runtime.decode = StreamDecodeState()
        return self
