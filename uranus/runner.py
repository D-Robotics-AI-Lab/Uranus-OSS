"""Single-GPU streaming inference runner for Uranus (v2 rig API).

``UranusRunner`` drives one autoregressive video generation session: it owns
the model replica (assembled lazily from a converted-weights directory), the
CPU preprocessing (image decode + robot skeleton rendering), and the session
state (KV cache, VAE causal-decoder cache, prompt context), which stays
resident on the GPU for the whole session.

Flow:

  * ``create()``: decode reference images, build the skeleton engine + camera
    rig, FK the reference qpos, compose the reference camera extrinsics,
    render the reference skeleton, encode the prompt (T5), VAE-encode the
    reference frames, prefill the DiT KV cache, then write the last reference
    latent straight into the KV slot (skipping the denoising loop) and decode
    it once to seed the VAE causal decoder cache. Emits no video.
  * ``step()``: ``num_step`` is rounded up to a multiple of the temporal
    interval (4 video frames == one latent frame); for each chunk it sets the
    engine state per frame, composes per-frame camera extrinsics from the rig
    (mounted cameras track their mount body via FK), renders the step
    skeleton, denoises one new latent frame, fills + trims the KV cache, and
    incrementally VAE-decodes 4 video frames. Returns per-camera uint8 frames.

Cameras and their mounts are configured once in the MJCF passed to ``create``;
steps carry joint positions, compact gripper state, and a robot transform. The
composed XML camera extrinsics are shared by the skeleton renderer and Plücker
conditioning.

Single-session semantics: ``create`` starts a new generation (an active
session is silently replaced), ``step`` advances it, ``close`` releases it.
The runner is single-threaded — callers must serialize. Models are assembled
on first use and stay loaded across generations; only the session state is
dropped by ``create``/``close``.

One runner = one fixed resolution (constructor ``height``/``width``) and one
GPU. Output frame size for ``step`` is exactly the model size ``(height,
width)`` regardless of the raw camera image sizes; intrinsics are scaled
between them internally (see ``uranus.utils.media``).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from uranus.skeleton import (
    CameraSpec,
    EESpec,
    RigSpec,
    SkeletonEngine,
    SkeletonSpec,
)
from uranus.utils.media import (
    decode_reference_images,
    pad_list,
    resolve_step_raw_sizes,
)
from uranus.utils.model_loader import load_models
from uranus.utils.video import plucker_to_rgb_frames, video_to_frames
from uranus.functional import UranusStreamConfig, compute_plucker_embeddings
from uranus.functional.stream_runner import (
    check_data_validity,
    decode_frame,
    fill_frame_kv_cache,
    generate_frame,
    get_prompt_context,
    initialize_stream_state,
    prefill,
)
from uranus.functional.stream_state import UranusStreamState
from uranus.skeleton.render import render_skeleton_frames


def parse_dtype(dtype_name: str) -> torch.dtype:
    """Map a dtype name (fp16/bf16/fp32) to a torch dtype."""
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype_name]
    except KeyError:
        raise ValueError(
            f"Unsupported dtype: {dtype_name!r} (expected one of {sorted(mapping)})"
        ) from None


def parse_frame(frame) -> np.ndarray | dict[str, Any]:
    """Normalize one joint/gripper/robot-transform frame."""
    if isinstance(frame, dict):
        fields = {"joint_positions", "gripper", "robot_transform"}
        legacy_fields = {"arm", "gripper", "robot_transform"}
        if any(field in frame for field in fields | {"arm"}):
            expected = legacy_fields if "arm" in frame else fields
            missing = sorted(expected - set(frame))
            extra = sorted(set(frame) - expected)
            if missing or extra:
                raise ValueError(
                    "state must contain exactly "
                    "joint_positions/gripper/robot_transform; "
                    f"missing={missing}, extra={extra}"
                )
            transform = frame["robot_transform"]
            if not isinstance(transform, dict):
                raise ValueError("robot_transform must be an object")
            return {
                "joint_positions": np.asarray(
                    frame.get("joint_positions", frame.get("arm")), dtype=np.float64
                )
                .reshape(-1)
                .tolist(),
                "gripper": np.asarray(frame["gripper"], dtype=np.float64)
                .reshape(-1)
                .tolist(),
                "robot_transform": {
                    "xyz": np.asarray(transform.get("xyz", []), dtype=np.float64)
                    .reshape(-1)
                    .tolist(),
                    "quaternion": np.asarray(
                        transform.get("quaternion", []), dtype=np.float64
                    )
                    .reshape(-1)
                    .tolist(),
                },
            }
        # Compatibility with the previous exported frame wrapper.
        return np.asarray(frame["observation.state"], dtype=np.float64)
    return np.asarray(frame, dtype=np.float64)


@dataclass
class _SessionMeta:
    """Session bookkeeping (replaces the legacy dict-based ``_meta``)."""

    camera_names: tuple[str, ...]
    engine: SkeletonEngine
    rig: RigSpec
    prompt: str
    seed: int
    generator: torch.Generator
    raw_sizes: dict[str, tuple[int, int]]
    scaled_intrinsics: dict[str, np.ndarray] = field(default_factory=dict)


class UranusRunner:
    """Single-GPU streaming Uranus inference runner (see module docstring)."""

    def __init__(
        self,
        converted_weights_dir: str,
        device: str | torch.device = "cuda:0",
        *,
        dtype: str = "bf16",
        height: int = 384,
        width: int = 640,
        num_inference_steps: int = 25,
        sigma_shift: float = 5.0,
        cfg_scale: float = 0.0,
        uranus_scale: float = 1.0,
        teacher_forcing_window_size: int = 2,
        skeleton_mode: str = "lightweight",
    ) -> None:
        self.device = torch.device(device)
        # Pin the process current device to the inference GPU. Without this,
        # running on a non-zero device (e.g. cuda:1) leaves torch.cuda's
        # current device at cuda:0, so all implicit-device CUDA objects land on
        # the wrong card.
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.weights_dir = converted_weights_dir
        self._dtype = parse_dtype(dtype)
        self._height = int(height)
        self._width = int(width)
        self._num_inference_steps = int(num_inference_steps)
        self._sigma_shift = float(sigma_shift)
        self._cfg_scale = float(cfg_scale)
        self._uranus_scale = float(uranus_scale)
        self._teacher_forcing_window_size = int(teacher_forcing_window_size)
        self._skeleton_mode = skeleton_mode

        self._models: dict[str, Any] | None = None
        self._state: UranusStreamState | None = None
        self._meta: _SessionMeta | None = None

    @property
    def models(self) -> dict[str, Any]:
        if self._models is None:
            self._models = load_models(
                self.weights_dir, self.device, self._dtype, skeleton_mode=self._skeleton_mode
            )
        return self._models

    def close(self) -> None:
        """Release the active session state (GPU tensors). Models stay loaded."""
        self._state = None
        self._meta = None

    # ---------------------------------------------------------------- create

    def create(
        self,
        *,
        prompt: str,
        mjcf_path: str,
        ref_cam_images: tuple[bytes, ...],
        ref_qpos: np.ndarray | dict[str, Any],
        cameras: list[CameraSpec] | tuple[CameraSpec, ...],
        end_effectors: list[EESpec] | tuple[EESpec, ...] | None = None,
        skeleton: SkeletonSpec | None = None,
        camera_names: tuple[str, ...] | None = None,
        seed: int = 1,
    ) -> None:
        """Start a new generation from the reference frame.

        ``ref_qpos`` contains joint positions, gripper, and robot transform
        state using the joint order declared by the MJCF, or a full qpos vector
        for compatibility.
        ``cameras`` is the ordered rig — its order fixes camera order for the
        whole session (KV streams, decode order, output naming).
        ``camera_names`` is accepted for backward compatibility and must match
        the rig names (order-insensitively checked as a set).
        """
        camera_names = tuple(camera_names) if camera_names is not None else tuple(c.name for c in cameras)
        if len(ref_cam_images) != len(cameras):
            raise ValueError(
                f"ref_cam_images count must equal rig size, "
                f"got {len(ref_cam_images)} images for {len(cameras)} cameras"
            )
        if set(camera_names) != {c.name for c in cameras}:
            raise ValueError(
                f"camera_names {camera_names} do not match rig names "
                f"{tuple(c.name for c in cameras)}"
            )

        rig = RigSpec(
            mjcf_path=mjcf_path,
            cameras=tuple(cameras),
            end_effectors=tuple(end_effectors) if end_effectors else (),
            skeleton=skeleton if skeleton is not None else SkeletonSpec(),
        )
        engine = SkeletonEngine(rig)

        qpos = parse_frame(ref_qpos)
        engine.set_state(qpos)

        target_size = (self._height, self._width)
        decoded = decode_reference_images(
            ref_images=ref_cam_images,
            camera_names=camera_names,
            target_size=target_size,
        )
        # raw sizes come from the decoded reference images; every camera's
        # scaled intrinsics are constant for the whole session.
        raw_sizes = decoded.raw_sizes
        scaled_intrinsics = {
            cam.name: engine.scaled_intrinsics(cam, raw_sizes[cam.name], target_size)
            for cam in rig.cameras
        }

        reference_camera_extrinsics = engine.compose_camera_extrinsics()
        ee_states = engine.get_ee_states()
        keypoints = engine.get_keypoints()
        sh_corrections = engine.get_ee_sh_corrections()
        reference_skeleton_images = [
            render_skeleton_frames(
                [ee_states],
                [keypoints],
                [reference_camera_extrinsics[cam_idx]],
                [scaled_intrinsics[camera.name]],
                target_size,
                sh_corrections=sh_corrections,
            )
            for cam_idx, camera in enumerate(rig.cameras)
        ]

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        state = self._create_stream_infer(
            prompt=prompt,
            reference_images=decoded.reference_images,
            reference_skeleton_images=[video[0] for video in reference_skeleton_images],
            reference_camera_extrinsics=reference_camera_extrinsics,
            reference_camera_intrinsics=[scaled_intrinsics[name] for name in camera_names],
            camera_names=camera_names,
            generator=generator,
        )

        self._state = state
        self._meta = _SessionMeta(
            camera_names=camera_names,
            engine=engine,
            rig=rig,
            prompt=prompt,
            seed=seed,
            generator=generator,
            raw_sizes=raw_sizes,
            scaled_intrinsics=scaled_intrinsics,
        )

    def _create_stream_infer(
        self,
        *,
        prompt: str,
        reference_images: list[torch.Tensor],
        reference_skeleton_images: list[torch.Tensor],
        reference_camera_extrinsics: list,
        reference_camera_intrinsics: list,
        camera_names: tuple[str, ...],
        generator: torch.Generator,
    ) -> UranusStreamState:
        models = self.models
        stream_config = self._build_stream_config(num_cameras=len(camera_names))
        # Create runs prefill + reference KV fill only — no denoising. The
        # returned first latents are the last reference frame; we still decode
        # it to seed the VAE's causal decoder cache so the first ``step``
        # decodes with the reference temporal context (the video itself is
        # discarded — create emits nothing until the first step).
        first_latents, state = self._run_stream(
            prompt,
            reference_images=reference_images,
            reference_skeleton_images=reference_skeleton_images,
            reference_camera_extrinsics=reference_camera_extrinsics,
            reference_camera_intrinsics=reference_camera_intrinsics,
            skeleton_images=[image.unsqueeze(0) for image in reference_skeleton_images],
            camera_extrinsics=[[cam_extr] for cam_extr in reference_camera_extrinsics],
            camera_intrinsics=[[cam_intr] for cam_intr in reference_camera_intrinsics],
            stream_config=stream_config,
            state=None,
            generator=generator,
        )
        with torch.no_grad():
            _, state = decode_frame(
                latents=first_latents,
                models=models,
                state=state,
                device=self.device,
                config=stream_config,
            )
        return state

    # ---------------------------------------------------------------- step

    def step(
        self,
        *,
        qpos: tuple[np.ndarray | dict[str, Any], ...],
        num_step: int,
        seed: int | None = None,
        return_skeleton: bool = False,
        return_plucker: bool = False,
    ) -> (
        dict[str, list[np.ndarray]]
        | tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]]]
        | tuple[
            dict[str, list[np.ndarray]],
            dict[str, list[np.ndarray]],
            dict[str, list[np.ndarray]],
        ]
    ):
        """Advance the session by ``num_step`` video frames.

        The frame count is rounded up to a multiple of the temporal interval
        (4); short sequences are padded by repeating the last frame's inputs.
        Each entry contains joint positions, gripper, and robot transform state
        (or is a legacy full-qpos vector). Returns per-camera RGB frames with
        exactly the rounded-up frame count. Optional visualization outputs are
        appended to the return tuple in skeleton, then Plücker order.
        """
        if self._state is None or self._meta is None:
            raise RuntimeError("create() must be called before step()")

        requested_steps = int(num_step)
        if requested_steps <= 0:
            raise ValueError("num_step must be > 0")
        if len(qpos) == 0:
            raise ValueError("qpos must not be empty")

        meta = self._meta
        engine = meta.engine
        camera_names = meta.camera_names
        if seed is not None and int(seed) != meta.seed:
            raise ValueError("seed is session-scoped; pass it to create()")

        chunk_size = int(self._state.temporal_interval)
        if chunk_size <= 0:
            raise RuntimeError(f"Invalid temporal_interval: {chunk_size}")
        actual_steps = ((requested_steps + chunk_size - 1) // chunk_size) * chunk_size

        qpos_sequence = [parse_frame(frame) for frame in qpos]
        qpos_sequence = pad_list(qpos_sequence, requested_steps, name="state")
        qpos_sequence = pad_list(qpos_sequence, actual_steps, name="state.chunk_align")

        target_size = (self._height, self._width)
        # The generator belongs to the session. Recreating it here used to
        # restart the same random sequence on every public step() call.
        generator = meta.generator

        stream_config = self._build_stream_config(num_cameras=len(camera_names))
        state = self._state
        output_chunks: list[torch.Tensor] = []
        skeleton_frames = {name: [] for name in camera_names}
        plucker_frames = {name: [] for name in camera_names}
        for start in range(0, actual_steps, chunk_size):
            end = start + chunk_size
            chunk_qpos = qpos_sequence[start:end]

            # FK every frame once, collect per-frame geometry, then compose
            # per-frame extrinsics per camera.
            ee_states_all, keypoints_all, extrinsics_all = [], [], []
            sh_corrections = None
            for frame_qpos in chunk_qpos:
                engine.set_state(frame_qpos)
                ee_states_all.append(engine.get_ee_states())
                keypoints_all.append(engine.get_keypoints())
                extrinsics_all.append(engine.compose_camera_extrinsics())
                if sh_corrections is None:
                    sh_corrections = engine.get_ee_sh_corrections()
            # extrinsics_all[t][cam] -> per-camera lists
            per_camera_extrinsics = [
                [extrinsics_all[t][cam_idx] for t in range(chunk_size)]
                for cam_idx in range(len(camera_names))
            ]

            skeleton_images: list[torch.Tensor] = []
            for cam_idx, camera in enumerate(engine.rig.cameras):
                K = meta.scaled_intrinsics[camera.name]
                video = render_skeleton_frames(
                    ee_states_all,
                    keypoints_all,
                    per_camera_extrinsics[cam_idx],
                    [K] * chunk_size,
                    target_size,
                    sh_corrections=sh_corrections,
                )
                skeleton_images.append(video)
                if return_skeleton:
                    skeleton_frames[camera.name].extend(
                        frame.permute(1, 2, 0).contiguous().cpu().numpy()
                        for frame in video
                    )
                if return_plucker:
                    plucker = compute_plucker_embeddings(
                        extrinsics=per_camera_extrinsics[cam_idx],
                        intrinsics=[K] * chunk_size,
                        height=target_size[0],
                        width=target_size[1],
                        device="cpu",
                    )
                    plucker_frames[camera.name].extend(
                        plucker_to_rgb_frames(plucker)
                    )

            camera_extrinsics = per_camera_extrinsics
            camera_intrinsics = [
                [meta.scaled_intrinsics[name]] * chunk_size for name in camera_names
            ]

            with torch.no_grad():
                generated_latents, state = self._run_stream(
                    meta.prompt,
                    reference_images=None,
                    reference_skeleton_images=None,
                    reference_camera_extrinsics=None,
                    reference_camera_intrinsics=None,
                    skeleton_images=skeleton_images,
                    camera_extrinsics=camera_extrinsics,
                    camera_intrinsics=camera_intrinsics,
                    stream_config=stream_config,
                    state=state,
                    generator=generator,
                )
                # The KV cache stays on the GPU for the whole step; it is never
                # moved off mid-loop (a mid-step relocation races the next
                # chunk's generation).
                if generated_latents is not None:
                    video, state = decode_frame(
                        latents=generated_latents,
                        models=self.models,
                        state=state,
                        device=self.device,
                        config=stream_config,
                    )
                    output_chunks.append(video)

        if not output_chunks:
            raise RuntimeError("_run_stream returned empty results")

        self._state = state
        video = torch.cat(output_chunks, dim=3).contiguous()  # dim 3 is time
        generated_frames = video_to_frames(video, camera_names)
        if return_skeleton and return_plucker:
            return generated_frames, skeleton_frames, plucker_frames
        if return_skeleton:
            return generated_frames, skeleton_frames
        if return_plucker:
            return generated_frames, plucker_frames
        return generated_frames

    # ---------------------------------------------------------------- internals

    def _build_stream_config(self, *, num_cameras: int) -> UranusStreamConfig:
        return UranusStreamConfig(
            num_cameras=num_cameras,
            height=self._height,
            width=self._width,
            num_inference_steps=self._num_inference_steps,
            sigma_shift=self._sigma_shift,
            cfg_scale=self._cfg_scale,
            uranus_scale=self._uranus_scale,
            teacher_forcing_window_size=self._teacher_forcing_window_size,
            skeleton_mode=self._skeleton_mode,
        )

    def _run_stream(
        self,
        prompt: str,
        *,
        reference_images,
        reference_skeleton_images,
        reference_camera_extrinsics,
        reference_camera_intrinsics,
        skeleton_images,
        camera_extrinsics,
        camera_intrinsics,
        stream_config,
        state,
        generator: torch.Generator,
    ):
        models = self.models
        with torch.no_grad():
            if state is None:  # create path: build state, then prefill
                state = initialize_stream_state(models, stream_config, self.device)
            state = get_prompt_context(prompt, models, state, self.device, self._dtype)
            if state.kv_cache is None:  # only the create path triggers this
                state, reference_latents, reference_fused_context = prefill(
                    reference_images=reference_images,
                    reference_skeleton_images=reference_skeleton_images,
                    reference_camera_extrinsics=reference_camera_extrinsics,
                    reference_camera_intrinsics=reference_camera_intrinsics,
                    models=models,
                    state=state,
                    device=self.device,
                    dtype=self._dtype,
                )
                # Skip generate_frame on create: the first generated frame is
                # essentially the reference latents, so write those straight
                # into the KV cache slot and skip the whole denoising loop.
                # The last reference frame comes back as ``first_latents`` so
                # the caller can decode it to seed the VAE causal cache (no
                # video is returned, but the decoder state must match the
                # first step's temporal context).
                #
                # NOTE: the ``[:, [0]].permute(0, 3, 2, 1, 4, 5)`` is
                # load-bearing — it moves the camera axis into the F slot of
                # the stream tensor the KV fill / incremental decode expect.
                # Do not "simplify" it into an equivalent rearrange.
                state = fill_frame_kv_cache(
                    generated_latents=reference_latents[:, [0]].permute(0, 3, 2, 1, 4, 5),
                    fused_context=reference_fused_context[:, [0]].permute(0, 3, 2, 1, 4, 5),
                    models=models,
                    state=state,
                    config=stream_config,
                )
                return reference_latents[:, [0]].permute(0, 3, 2, 1, 4, 5), state
            if skeleton_images is None:
                return None, state
            check_data_validity(skeleton_images, camera_extrinsics, camera_intrinsics, state)
            generated_latents, fused_context = generate_frame(
                skeleton_images=skeleton_images,
                camera_extrinsics=camera_extrinsics,
                camera_intrinsics=camera_intrinsics,
                models=models,
                state=state,
                device=self.device,
                dtype=self._dtype,
                config=stream_config,
                generator=generator,
            )
            state = fill_frame_kv_cache(
                generated_latents=generated_latents,
                fused_context=fused_context,
                models=models,
                state=state,
                config=stream_config,
            )
            return generated_latents, state
