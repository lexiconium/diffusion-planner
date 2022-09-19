import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput

from ..configuration_utils import ConfigUtilsMixin, arguments_to_config


def get_timestep_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.

    :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = torch.exp(exponent).to(device=timesteps.device)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    @arguments_to_config
    def __init__(self, time_embedding_dim: int):
        super().__init__()

        self.time_embedding_dim = time_embedding_dim

        self.learnable_adapter = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 4),
            nn.Mish(),
            nn.Linear(time_embedding_dim * 4, time_embedding_dim)
        )

    def forward(
        self,
        timesteps: torch.Tensor,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float
    ):
        time_embed = get_timestep_embedding(
            timesteps,
            embedding_dim=self.time_embedding_dim,
            flip_sin_to_cos=flip_sin_to_cos,
            downscale_freq_shift=downscale_freq_shift
        )
        time_embed = self.learnable_adapter(time_embed)

        return time_embed


class TemporalResnetBlock(nn.Module):
    @arguments_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        time_embed_channels: int,
        eps: float = 1e-5,
        dropout: float = 0.0
    ):
        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=eps),
            nn.Mish(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.time_embed_proj = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embed_channels, out_channels)
        )

        self.conv_block_2 = nn.Sequential(
            nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, eps=eps),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else
            nn.Identity()
        )

    def forward(self, hidden_states: torch.Tensor, time_embedding: torch.Tensor):
        _hidden_states = hidden_states

        _hidden_states = self.conv_block_1(_hidden_states)
        time_embedding = self.time_embed_proj(time_embedding)[:, :, None]
        _hidden_states = _hidden_states + time_embedding

        _hidden_states = self.conv_block_2(_hidden_states)

        return _hidden_states + self.residual_conv(hidden_states)


class TemporalUnetBlock(nn.Module):
    @arguments_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        time_embed_channels: int,
        num_layers: int = 2,
        eps: float = 1e-5,
        dropout: float = 0.0
    ):
        super().__init__()

        self.resnets = nn.ModuleList([])

        for i in range(num_layers):
            self.resnets.append(
                TemporalResnetBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    time_embed_channels=time_embed_channels,
                    eps=eps,
                    num_groups=num_groups,
                    dropout=dropout
                )
            )

    def forward(self, hidden_states: torch.Tensor, time_embedding: torch.Tensor):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, time_embedding)

        return hidden_states


class TemporalUnetDownBlock(nn.Module):
    @arguments_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        time_embed_channels: int,
        num_layers: int = 2,
        eps: float = 1e-5,
        dropout: float = 0.0,
        add_downsampler: bool = True
    ):
        super().__init__()

        self.resnets = nn.ModuleList([])

        for i in range(num_layers):
            self.resnets.append(
                TemporalResnetBlock(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    time_embed_channels=time_embed_channels,
                    eps=eps,
                    num_groups=num_groups,
                    dropout=dropout
                )
            )

        self.downsampler = None
        if add_downsampler:
            self.downsampler = nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1
            )

    def forward(self, hidden_states: torch.Tensor, time_embedding: torch.Tensor):
        skipped_hidden_states_tuple = ()

        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, time_embedding)
            skipped_hidden_states_tuple += (hidden_states,)

        if self.downsampler is not None:
            hidden_states = self.downsampler(hidden_states)

        return hidden_states, skipped_hidden_states_tuple


class TemporalUnetUpBlock(nn.Module):
    @arguments_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        time_embed_channels: int,
        num_layers: int = 2,
        eps: float = 1e-5,
        dropout: float = 0.0,
        add_upsampler: bool = True
    ):
        super().__init__()

        self.resnets = nn.ModuleList([])

        for i in range(num_layers):
            self.resnets.append(
                TemporalResnetBlock(
                    in_channels=(in_channels if i == 0 else out_channels) + out_channels,
                    out_channels=out_channels,
                    time_embed_channels=time_embed_channels,
                    eps=eps,
                    num_groups=num_groups,
                    dropout=dropout
                )
            )

        self.upsampler = None
        if add_upsampler:
            self.upsampler = nn.ConvTranspose1d(
                out_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        skipped_hidden_states_tuple: Tuple[torch.Tensor, ...],
        time_embedding: torch.Tensor
    ):
        for resnet, skipped_hidden_states in zip(self.resnets, skipped_hidden_states_tuple[::-1]):
            hidden_states = torch.cat([hidden_states, skipped_hidden_states], dim=1)
            hidden_states = resnet(hidden_states, time_embedding)

        if self.upsampler is not None:
            hidden_states = self.upsampler(hidden_states)

        return hidden_states


@dataclass
class TemporalUnetOutput(ModelOutput):
    sample: torch.Tensor


class TemporalUnet(nn.Module, ConfigUtilsMixin):
    @arguments_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_groups: int,
        num_layers_per_block: int = 2,
        norm_eps: float = 1e-5,
        dropout: float = 0.0
    ):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        time_embedding_dim = block_out_channels[0]
        self.time_embedding = TimestepEmbedding(time_embedding_dim)

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        _in_channels = block_out_channels[0]
        for i, _out_channels in enumerate(block_out_channels):
            is_final_block = i == len(block_out_channels) - 1

            self.down_blocks.append(
                TemporalUnetDownBlock(
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    time_embed_channels=time_embedding_dim,
                    num_layers=num_layers_per_block,
                    eps=norm_eps,
                    num_groups=num_groups,
                    dropout=dropout,
                    add_downsampler=not is_final_block
                )
            )

            _in_channels = _out_channels

        self.mid_block = TemporalUnetBlock(
            in_channels=block_out_channels[-1],
            out_channels=block_out_channels[-1],
            time_embed_channels=time_embedding_dim,
            num_layers=num_layers_per_block,
            eps=norm_eps,
            num_groups=num_groups,
            dropout=dropout
        )

        for i, _out_channels in enumerate(block_out_channels[::-1]):
            is_final_block = i == len(block_out_channels) - 1

            self.up_blocks.append(
                TemporalUnetUpBlock(
                    in_channels=_in_channels,
                    out_channels=_out_channels,
                    time_embed_channels=time_embedding_dim,
                    num_layers=num_layers_per_block,
                    eps=norm_eps,
                    num_groups=num_groups,
                    dropout=dropout,
                    add_upsampler=not is_final_block
                )
            )

            _in_channels = _out_channels

        self.conv_out = nn.Sequential(
            nn.GroupNorm(
                num_groups=num_groups, num_channels=block_out_channels[0], eps=norm_eps
            ),
            nn.Mish(),
            nn.Conv1d(block_out_channels[0], out_channels, kernel_size=5, padding=2)
        )

    def forward(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,  # goal conditioning?
        flip_sin_to_cos: bool = False,
        downscale_freq_shift: float = 1.0
    ):
        time_embedding = self.time_embedding(
            timesteps, flip_sin_to_cos=flip_sin_to_cos, downscale_freq_shift=downscale_freq_shift
        )

        # preprocess
        sample = sample.transpose(1, 2)
        sample = self.conv_in(sample)

        # downsample
        skipped_hidden_states_tuple = (sample,)
        for down_block in self.down_blocks:
            sample, _skipped_hidden_states_tuple = down_block(sample, time_embedding=time_embedding)
            skipped_hidden_states_tuple += _skipped_hidden_states_tuple

        # mid
        sample = self.mid_block(sample, time_embedding=time_embedding)

        # upsample
        for up_block in self.up_blocks:
            _skipped_hidden_states_tuple = skipped_hidden_states_tuple[-len(up_block.resnets):]
            skipped_hidden_states_tuple = skipped_hidden_states_tuple[:-len(up_block.resnets)]
            sample = up_block(sample, _skipped_hidden_states_tuple, time_embedding=time_embedding)

        # postprocess
        sample = self.conv_out(sample)
        sample = sample.transpose(1, 2)

        return TemporalUnetOutput(sample=sample)
