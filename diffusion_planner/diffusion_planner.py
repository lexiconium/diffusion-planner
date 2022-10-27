import inspect
from typing import Union

import numpy as np
import torch
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from tqdm.auto import tqdm

from .models.trajectory_unet import TrajectoryUNet


class DiffusionPlanner:
    def __init__(self, unet: TrajectoryUNet):
        self.unet = unet

    @torch.no_grad()
    def __call__(
        self,
        observations: Union[np.ndarray, list[np.ndarray]],
        noise_scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        target_horizon: int,
        num_inference_steps: int = 50,
        return_diffusion_steps: bool = False,
        **kwargs
    ):
        # Check types
        if isinstance(observations, np.ndarray):
            observations = [observations]
        if not isinstance(observations, (list, tuple)) or not isinstance(observations[0], np.ndarray):
            raise ValueError("Observations must be a numpy array or a list of numpy arrays.")

        # Make batched
        observations = torch.stack(
            [torch.as_tensor(observation, device=self.device) for observation in observations],
            dim=0
        )

        batch_size, observation_dim = observations.shape

        samples = torch.randn(
            (batch_size, target_horizon, self.unet.conv_in.in_channels),
            device=self.device
        )

        noise_scheduler.set_timesteps(num_inference_steps)

        diffusion_steps = noise_scheduler.timesteps.to(self.device)

        # Scale the initial noise by the standard deviation required by the scheduler
        samples = samples * noise_scheduler.init_noise_sigma

        # Prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        extra_step_kwargs = {}

        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(noise_scheduler.step).parameters.keys())
        if accepts_eta and "eta" in kwargs:
            extra_step_kwargs["eta"] = kwargs.pop("eta")

        self.unet.to(self.device)

        for i, t in enumerate(tqdm(diffusion_steps)):
            # Fix initial observation
            samples[:, 0, :observation_dim] = observations

            samples = noise_scheduler.scale_model_input(samples, t)

            # Predict noise residual
            noise_predictions = self.unet(samples, t).samples

            samples = noise_scheduler.step(noise_predictions, t, samples, **extra_step_kwargs).prev_sample

        samples[:, 0, :observation_dim] = observations
        samples = samples.cpu().numpy()

        return samples

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
