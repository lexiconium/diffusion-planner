from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from transformers.modeling_outputs import ModelOutput

from ..modeling_utils import torch_float_or_long
from ..temporal_unet import TemporalUnet


@dataclass
class DiffusionPlannerTrainingModelOutput(ModelOutput):
    loss: torch.Tensor


class DiffusionPlannerTrainingModel(nn.Module):
    def __init__(self, scheduler: DDPMScheduler, unet: TemporalUnet):
        super().__init__()

        self.scheduler = scheduler
        self.unet = unet

    @torch_float_or_long
    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ):
        transitions = torch.cat([observations, actions], dim=-1)

        noise = torch.randn_like(transitions, device=transitions.device)
        timesteps = torch.randint(
            len(self.scheduler),
            size=(transitions.shape[0],),
            dtype=torch.long,
            device=transitions.device
        )

        observation_dim = observations.shape[-1]
        constraint = observations[:, 0]

        noisy_transitions = self.scheduler.add_noise(transitions, noise, timesteps)
        noisy_transitions[..., 0, :observation_dim] = constraint
        noisy_transitions.masked_fill_(masks[..., None], 0)

        noise_pred = self.unet(noisy_transitions, timesteps, context).sample
        noise_pred[..., 0, :observation_dim] = constraint
        noise_pred.masked_fill_(masks[..., None], 0)

        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean([1, 2]).mean()

        return DiffusionPlannerTrainingModelOutput(loss=loss)
