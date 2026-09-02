"""Single-GPU streaming inference runner for Uranus.

``UranusRunner`` drives one autoregressive video generation session: it owns
the model replica (assembled lazily from a converted-weights directory), the
CPU preprocessing (image decode + robot skeleton rendering), and the session
state (KV cache, VAE causal-decoder cache, prompt context), which stays
resident on the GPU for the whole session.

Flow:

  * ``create()``: decode reference images, render the reference skeleton,
    encode the prompt (T5), VAE-encode the reference frames, prefill the DiT
    KV cache, then write the last reference latent straight into the KV slot
    (skipping the denoising loop) and decode it once to seed the VAE causal
    decoder cache. Emits no video.
  * ``step()``: ``num_step`` is rounded up to a multiple of the temporal
    interval (4 video frames == one latent frame); for each chunk it renders
    the step skeleton, denoises one new latent frame (``num_inference_steps``
    sigma steps), fills + trims the KV cache, and incrementally VAE-decodes
    4 video frames. Returns per-camera uint8 frames.

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

from typing import Any

import numpy as np
import torch

from uranus.utils.media import (
    decode_reference_images,
    pad_step_inputs,
    render_reference_skeleton,
    render_step_skeleton_chunk,
    require_robot_obs,
    resolve_step_raw_sizes,
    scale_camera_intrinsics,
)
from uranus.utils.model_loader import load_models
from uranus.utils.video import video_to_frames
from uranus.functional import UranusStreamConfig
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
        self._meta: dict[str, Any] | None = None  # session bookkeeping, see create

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

    def create(
        self,
        *,
        prompt: str,
        ref_cam_images: tuple[bytes, ...],
        ref_cam_extrinsics: dict[str, np.ndarray],
        ref_cam_intrinsics: dict[str, np.ndarray],
        ref_qpos: np.ndarray | dict[str, Any],
        ref_robot_obs: np.ndarray | dict[str, Any] | None = None,
        robot_type: str,
        camera_names: tuple[str, ...],
        seed: int = 1,
    ) -> None:
        """Start a new generation from the reference frame.

        Encodes the prompt, prefills the DiT KV cache from the reference
        images and seeds the VAE causal decoder cache. No video is produced
        here — the first video frames come from ``step``. An already-active
        session is replaced.

        ``ref_qpos`` may be a bare joint vector or a full telemetry frame dict
        (``{"observation.state": ..., "observation.robot": ...}``); in the
        latter case ``ref_robot_obs`` defaults to the embedded
        ``observation.robot`` payload.
        """
        if len(ref_cam_images) != len(camera_names):
            raise ValueError(
                f"ref_cam_images count must equal camera_names count, "
                f"got {len(ref_cam_images)} images for {len(camera_names)} cameras"
            )
        if isinstance(ref_qpos, dict):
            if ref_robot_obs is None:
                ref_robot_obs = ref_qpos.get("observation.robot")
            ref_qpos = ref_qpos["observation.state"]
        require_robot_obs(robot_type, ref_robot_obs, field_name="ref_robot_obs")

        target_size = (self._height, self._width)

        decoded = decode_reference_images(
            ref_images=ref_cam_images,
            calib_ext=ref_cam_extrinsics,
            calib_int=ref_cam_intrinsics,
            camera_names=camera_names,
            target_size=target_size,
        )
        reference_skeleton_images = render_reference_skeleton(
            robot_type=robot_type,
            ref_qpos=ref_qpos,
            ref_robot_obs=ref_robot_obs,
            decoded=decoded,
            camera_names=camera_names,
            target_size=target_size,
        )

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        state = self._create_stream_infer(
            prompt=prompt,
            reference_images=decoded.reference_images,
            reference_skeleton_images=reference_skeleton_images,
            reference_camera_extrinsics=decoded.reference_camera_extrinsics,
            reference_camera_intrinsics=decoded.reference_camera_intrinsics,
            camera_names=camera_names,
            generator=generator,
        )

        self._state = state
        self._meta = {
            "camera_names": camera_names,
            "robot_type": robot_type,
            "prompt": prompt,
            "seed": seed,
            "raw_sizes": decoded.raw_sizes,
        }

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


    def step(
        self,
        *,
        qpos: tuple[np.ndarray, ...] | tuple[dict[str, Any], ...],
        cam_extrinsics: dict[str, list[np.ndarray]],
        cam_intrinsics: dict[str, list[np.ndarray]],
        num_step: int,
        robot_obs: tuple[Any, ...] | None = None,
        seed: int | None = None,
    ) -> dict[str, list[np.ndarray]]:
        """Advance the session by ``num_step`` video frames.

        The frame count is rounded up to a multiple of the temporal interval
        (4); short sequences are padded by repeating the last frame's inputs.
        Returns ``{camera: [H, W, 3] uint8 RGB frames]}`` with exactly the
        rounded-up frame count per camera.
        """
        if self._state is None or self._meta is None:
            raise RuntimeError("create() must be called before step()")

        requested_steps = int(num_step)
        if requested_steps <= 0:
            raise ValueError("num_step must be > 0")
        if len(qpos) == 0:
            raise ValueError("qpos must not be empty")

        meta = self._meta
        camera_names: tuple[str, ...] = meta["camera_names"]
        seed = int(seed) if seed is not None else meta["seed"]

        # Chunk alignment: one latent frame covers ``temporal_interval`` video
        # frames (VAE temporal stride). Round up and repeat-last the inputs.
        chunk_size = int(self._state.temporal_interval)
        if chunk_size <= 0:
            raise RuntimeError(f"Invalid temporal_interval: {chunk_size}")
        actual_steps = ((requested_steps + chunk_size - 1) // chunk_size) * chunk_size

        # ``qpos`` entries may be bare joint vectors or full telemetry frame
        # dicts ({"observation.state": ..., "observation.robot": ...}) — the
        # latter contribute their robot observation unless ``robot_obs`` is
        # given explicitly.
        robot_obs_sequence = list(robot_obs) if robot_obs is not None else None
        qpos_sequence: list[np.ndarray] = []
        if robot_obs_sequence is None:
            split = [to_qpos_frame(u) for u in qpos]
            qpos_sequence = [state for state, _obs in split]
            if split and all(obs is not None for _, obs in split):
                robot_obs_sequence = [obs for _, obs in split]
        else:
            qpos_sequence = [to_qpos_state(u) for u in qpos]
        # pad_step_inputs mutates the per-camera dicts in place — pad copies so
        # the caller's data is untouched.
        camera_extrinsics = {camera: list(ext) for camera, ext in cam_extrinsics.items()}
        camera_intrinsics = {camera: list(intr) for camera, intr in cam_intrinsics.items()}

        raw_sizes = resolve_step_raw_sizes(camera_names, self._height, self._width, meta["raw_sizes"])
        target_size = (self._height, self._width)
        scaled_camera_intrinsics = scale_camera_intrinsics(
            camera_names=camera_names,
            camera_intrinsics=camera_intrinsics,
            raw_sizes=raw_sizes,
            target_size=target_size,
        )
        qpos_sequence, robot_obs_sequence = pad_step_inputs(
            camera_names=camera_names,
            qpos_sequence=qpos_sequence,
            robot_obs_sequence=robot_obs_sequence,
            camera_extrinsics=camera_extrinsics,
            camera_intrinsics=camera_intrinsics,
            scaled_camera_intrinsics=scaled_camera_intrinsics,
            requested_steps=requested_steps,
            actual_steps=actual_steps,
        )
        require_robot_obs(meta["robot_type"], robot_obs_sequence, field_name="robot_obs")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        stream_config = self._build_stream_config(num_cameras=len(camera_names))
        state = self._state
        output_chunks: list[torch.Tensor] = []
        for start in range(0, actual_steps, chunk_size):
            end = start + chunk_size
            skeleton_images, calib = render_step_skeleton_chunk(
                qpos_chunk=qpos_sequence[start:end],
                robot_obs_chunk=robot_obs_sequence[start:end]
                if robot_obs_sequence is not None
                else None,
                camera_names=camera_names,
                camera_extrinsics=camera_extrinsics,
                camera_intrinsics=camera_intrinsics,
                scaled_camera_intrinsics=scaled_camera_intrinsics,
                raw_sizes=raw_sizes,
                target_size=target_size,
                robot_type=meta["robot_type"],
                start=start,
                chunk_size=chunk_size,
            )
            with torch.no_grad():
                generated_latents, state = self._run_stream(
                    meta["prompt"],
                    reference_images=None,
                    reference_skeleton_images=None,
                    reference_camera_extrinsics=None,
                    reference_camera_intrinsics=None,
                    skeleton_images=[torch.stack(per_cam) for per_cam in skeleton_images],
                    camera_extrinsics=calib.extrinsics_list,  # scaled → model
                    camera_intrinsics=calib.intrinsics_list,  # scaled → model
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
        return video_to_frames(video, camera_names)


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


def to_qpos_state(frame: np.ndarray | dict[str, Any]) -> np.ndarray:
    """Extract the joint-state vector from a telemetry frame.

    Accepts either a bare ``np.ndarray`` joint vector or a frame dict shaped
    like the dataset's ``{"observation.state": ndarray, ...}`` records.
    """
    if isinstance(frame, dict):
        return frame["observation.state"]
    return frame


def to_qpos_frame(frame) -> tuple[np.ndarray, Any | None]:
    """Split one telemetry frame into ``(joint_state, robot_obs)``.

    Bare joint vectors yield ``(vector, None)``; dict frames shaped like the
    dataset's ``{"observation.state": ..., "observation.robot": ...}`` records
    yield their ``observation.robot`` payload (or None when absent).
    """
    if isinstance(frame, dict):
        return frame["observation.state"], frame.get("observation.robot")
    return frame, None
