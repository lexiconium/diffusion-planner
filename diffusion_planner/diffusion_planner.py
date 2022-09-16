from dataclasses import dataclass
from typing import List, Union

import torch
from diffusers import DDIMScheduler, DDPMScheduler, LMSDiscreteScheduler, PNDMScheduler
from tqdm.auto import tqdm
from transformers.modeling_outputs import ModelOutput

from .models.modeling_utils import torch_float_or_long
from .models.temporal_unet import TemporalUnet


@dataclass
class DiffusionPlannerOutput(ModelOutput):
    sample: torch.Tensor


@dataclass
class DiffusionPlanner:
    unet: TemporalUnet

    @torch_float_or_long
    def __call__(
        self,
        observations: Union[torch.Tensor, List[torch.Tensor]],
        scheduler: Union[DDPMScheduler, DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        horizon: int,
        num_inference_steps: int = 50
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

        # set initial observation
        sample[:, :1, :observation_dim] = observations  # TODO: trajectory steps conditioning

        scheduler.set_timesteps(num_inference_steps)

        for t in tqdm(scheduler.timesteps):
            timesteps = torch.full((batch_size,), t)
            noise_pred = self.unet(sample, timesteps).sample
            sample = scheduler.step(noise_pred, t, sample).prev_sample

        return DiffusionPlannerOutput(sample=sample)
