import inspect
from dataclasses import dataclass
from typing import List, Union

import torch
from diffusers import DDIMScheduler, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import BaseOutput
from tqdm.auto import tqdm

from .models.temporal_unet import TemporalUnet


@dataclass
class DiffusionPlannerOutput(BaseOutput):
    sample: torch.Tensor


@dataclass
class DiffusionPlanner:
    unet: TemporalUnet

    def __call__(
        self,
        observations: Union[torch.Tensor, List[torch.Tensor]],
        scheduler: Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        horizon: int,
        num_inference_steps: int = 50,
        **kwargs
    ):
        def make_batched(_observations: Union[torch.Tensor, List[torch.Tensor]]):
            num_dims = len(_observations.shape)

            if isinstance(_observations, torch.Tensor):
                if num_dims == 1:
                    return _observations[None, None, :]
                if num_dims == 2:
                    return _observations[:, None, :]
                if num_dims == 3:
                    return _observations

                raise ValueError(f"Unexpected observations shape {_observations.shape}.")

            return torch.cat([make_batched(_observation) for _observation in _observations], dim=0)

        observations = make_batched(observations)
        batch_size, trajectory_steps, observation_dim = observations.shape

        sample = torch.randn(
            (batch_size, horizon, self.unet.config.in_channels),
            device=observations.device
        )

        set_timesteps_kwargs = {
            kw: arg for kw, arg in kwargs.items()
            if kw in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        }
        scheduler.set_timesteps(num_inference_steps, **set_timesteps_kwargs)

        if isinstance(scheduler, LMSDiscreteScheduler):
            sample = sample * scheduler.sigmas[0]

        # set initial observation
        sample[:, :1, :observation_dim] = observations[:, :1, :]  # TODO: multi trajectory step conditioning

        step_kwargs = {
            kw: arg for kw, arg in kwargs.items()
            if kw in set(inspect.signature(scheduler.step).parameters.keys())
        }

        for i, t in enumerate(tqdm(scheduler.timesteps)):
            timesteps = torch.full((batch_size,), t)
            noise_pred = self.unet(sample, timesteps).sample
            if isinstance(scheduler, LMSDiscreteScheduler):
                sample = scheduler.step(noise_pred, i, sample, **step_kwargs).prev_sample
            else:
                sample = scheduler.step(noise_pred, t, sample, **step_kwargs).prev_sample

        return DiffusionPlannerOutput(sample=sample)
