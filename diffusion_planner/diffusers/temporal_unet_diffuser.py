from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import nn
from transformers.modeling_outputs import ModelOutput

from ..models.temporal_unet import TemporalUnet


@dataclass
class TemporalUnetDiffuserOutput(ModelOutput):
    loss: torch.Tensor


class TemporalUnetDiffuserForDDPM(nn.Module):
    def __init__(self, unet: TemporalUnet, scheduler: DDPMScheduler):
        super().__init__()

        self.unet = unet
        self.scheduler = scheduler

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
        # TODO: add make batched

        observation_dim = observations.shape[-1]
        constraint = observations[:, 0]

        noisy_transitions = self.scheduler.add_noise(transitions, noise, timesteps)
        noisy_transitions[..., 0, :observation_dim] = constraint
        noisy_transitions.masked_fill_(masks[..., None].bool(), 0)

        noise_pred = self.unet(noisy_transitions, timesteps, context).sample
        noise_pred[..., 0, :observation_dim] = constraint
        noise_pred.masked_fill_(masks[..., None].bool(), 0)

        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction="none").mean([1, 2]).mean()

        return TemporalUnetDiffuserOutput(loss=loss)
