"""
Flow Matching Scheduler for WanVideo inference (Euler ODE solver).

Migrated from diffsynth.diffusion.flow_match.FlowMatchScheduler, keeping only
the inference-relevant methods. Training-only methods (add_noise, training_target,
training_weight, etc.) are omitted.
"""

import torch

class FlowMatchScheduler:
    """
    WanVideo flow matching scheduler with sigma-shifted timestep schedule.

    sigma_shift warps the linear schedule to concentrate steps near sigma=1
    (noisier), improving sample quality. Default shift=5.0 for WanVideo.
    """

    def __init__(self):
        self.num_train_timesteps = 1000
        self.sigmas: torch.Tensor | None = None
        self.timesteps: torch.Tensor | None = None

    def set_timesteps(
        self,
        num_inference_steps: int = 50,
        denoising_strength: float = 1.0,
        shift: float = 5.0,
        device: torch.device = torch.device("cpu"),
    ):
        sigma_min = 0.0
        sigma_max = 1.0
        sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
        sigmas = torch.linspace(sigma_start, sigma_min, num_inference_steps + 1, device=device)[:-1]
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = sigmas * self.num_train_timesteps
        self.sigmas = sigmas

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        to_final: bool = False,
    ) -> torch.Tensor:
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]
        if to_final:
            sigma_next = 0
        else:
            sigma_next = self.sigmas[timestep_id + 1]
        return sample + model_output * (sigma_next - sigma)
