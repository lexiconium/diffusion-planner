import math
from typing import Optional

import torch
from torch import nn
from transformers.modeling_outputs import ModelOutput


class SinusoidalPositionalEncoding:
    def __init__(self, encoding_dim: int, max_position: int = 10000):
        self.encoding_dim = encoding_dim
        self.max_position = max_position

    def __call__(self, positions: torch.LongTensor) -> torch.Tensor:
        half_dim = self.encoding_dim // 2

        freq_ratios = torch.arange(half_dim, dtype=torch.float, device=positions.device) / (half_dim - 1)
        unit_encoding = torch.exp(-math.log(self.max_position) * freq_ratios)

        encodings = positions[:, None].float() * unit_encoding[None, :]
        encodings = torch.cat([torch.sin(encodings), torch.cos(encodings)], dim=-1)

        if self.encoding_dim % 2:
            encodings = torch.nn.functional.pad(encodings, (0, 1))

        return encodings


class DiffusionStepEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super(DiffusionStepEmbedding, self).__init__()

        self.embedding_dim = embedding_dim

        self.encodings = SinusoidalPositionalEncoding(embedding_dim)

        self.embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.Mish(),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )

    def forward(self, diffusion_steps: torch.LongTensor) -> torch.Tensor:
        encodings = self.encodings(diffusion_steps)
        embeddings = self.embedding(encodings)

        return embeddings


class TrajectoryConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, num_groups: int):
        super(TrajectoryConv1d, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(num_groups, num_channels=out_channels),
            nn.Mish()
        )

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        return self.conv(hidden_states)


class TrajectoryResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        diffusion_step_embedding_dim: int,
        kernel_size: int,
        num_groups: int
    ):
        super(TrajectoryResNetBlock, self).__init__()

        self.conv_block_1 = TrajectoryConv1d(
            in_channels, out_channels, kernel_size=kernel_size, num_groups=num_groups
        )

        self.diffusion_step_embedding_projector = nn.Sequential(
            nn.Mish(),
            nn.Linear(diffusion_step_embedding_dim, out_channels)
        )

        self.conv_block_2 = TrajectoryConv1d(
            out_channels, out_channels, kernel_size=kernel_size, num_groups=num_groups
        )

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else
            nn.Identity()
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        diffusion_step_embedding: torch.FloatTensor
    ) -> torch.FloatTensor:
        outputs = self.conv_block_1(hidden_states)
        diffusion_step_embedding = self.diffusion_step_embedding_projector(diffusion_step_embedding)

        outputs = outputs + diffusion_step_embedding[..., None]

        outputs = self.conv_block_2(outputs)
        residuals = self.residual_conv(hidden_states)

        outputs = outputs + residuals

        return outputs


class TrajectoryUNetBlockOutput(ModelOutput):
    samples: torch.FloatTensor
    skip_connection_samples: Optional[torch.FloatTensor]


class TrajectoryUNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        diffusion_step_embedding_dim: int,
        kernel_size: int,
        num_groups: int,
        num_layers: int = 2
    ):
        super(TrajectoryUNetBlock, self).__init__()

        self.resnets = nn.ModuleList([
            TrajectoryResNetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                diffusion_step_embedding_dim=diffusion_step_embedding_dim,
                kernel_size=kernel_size,
                num_groups=num_groups
            )
            for i in range(num_layers)
        ])

    def forward(
        self,
        samples: torch.FloatTensor,
        diffusion_step_embedding: torch.FloatTensor,
        skip_connection_samples: Optional[torch.FloatTensor] = None
    ) -> TrajectoryUNetBlockOutput:
        for resnet in self.resnets:
            samples = resnet(samples, diffusion_step_embedding)

        return TrajectoryUNetBlockOutput(
            samples=samples,
            skip_connection_samples=None
        )


class TrajectoryUNetDownBlock(TrajectoryUNetBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        diffusion_step_embedding_dim: int,
        kernel_size: int,
        num_groups: int,
        num_layers: int = 2,
        add_downsampler: bool = True
    ):
        super(TrajectoryUNetDownBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            diffusion_step_embedding_dim=diffusion_step_embedding_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            num_layers=num_layers
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

    def forward(
        self,
        samples: torch.FloatTensor,
        diffusion_step_embedding: torch.FloatTensor,
        skip_connection_samples: Optional[torch.FloatTensor] = None
    ) -> TrajectoryUNetBlockOutput:

        for resnet in self.resnets:
            samples = resnet(samples, diffusion_step_embedding)

        skip_connection_samples = samples

        if self.downsampler is not None:
            samples = self.downsampler(samples)

        return TrajectoryUNetBlockOutput(
            samples=samples,
            skip_connection_samples=skip_connection_samples
        )


class TrajectoryUNetUpBlock(TrajectoryUNetBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        diffusion_step_embedding_dim: int,
        kernel_size: int,
        num_groups: int,
        num_layers: int = 2,
        add_upsampler: bool = True
    ):
        super(TrajectoryUNetUpBlock, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            diffusion_step_embedding_dim=diffusion_step_embedding_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            num_layers=num_layers
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
        samples: torch.FloatTensor,
        diffusion_step_embedding: torch.FloatTensor,
        skip_connection_samples: Optional[torch.FloatTensor] = None
    ) -> TrajectoryUNetBlockOutput:
        samples = torch.cat([samples, skip_connection_samples], dim=1)

        for resnet in self.resnets:
            samples = resnet(samples, diffusion_step_embedding)

        skip_connection_samples = samples

        if self.upsampler is not None:
            samples = self.upsampler(samples)

        return TrajectoryUNetBlockOutput(
            samples=samples,
            skip_connection_samples=skip_connection_samples
        )


class TrajectoryUNetOutput(ModelOutput):
    samples: torch.FloatTensor


class TrajectoryUNet(nn.Module):
    def __init__(
        self,
        transition_dim: int,
        intermediate_dims: tuple[int, ...],
        kernel_size: int,
        num_groups: int,
        num_layers: int
    ):
        super(TrajectoryUNet, self).__init__()

        diffusion_step_embedding_dim = intermediate_dims[0]
        self.diffusion_step_embedding = DiffusionStepEmbedding(diffusion_step_embedding_dim)

        self.conv_in = nn.Conv1d(transition_dim, intermediate_dims[0], kernel_size=3, padding=1)

        self.downsample_blocks = nn.ModuleList([])
        self.mid_block = None
        self.upsample_blocks = nn.ModuleList([])

        in_channels = intermediate_dims[0]

        for i, out_channels in enumerate(intermediate_dims[1:]):
            is_last_block = i == len(intermediate_dims) - 2

            self.downsample_blocks.append(
                TrajectoryUNetDownBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    diffusion_step_embedding_dim=diffusion_step_embedding_dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    num_layers=num_layers,
                    add_downsampler=not is_last_block
                )
            )

            in_channels = out_channels

        self.mid_block = TrajectoryUNetBlock(
            in_channels=intermediate_dims[-1],
            out_channels=intermediate_dims[-1],
            diffusion_step_embedding_dim=diffusion_step_embedding_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            num_layers=num_layers,
        )

        for i, out_channels in enumerate(intermediate_dims[:-1][::-1]):
            is_last_block = i == len(intermediate_dims) - 2

            self.upsample_blocks.append(
                TrajectoryUNetUpBlock(
                    in_channels=2 * in_channels,
                    out_channels=out_channels,
                    diffusion_step_embedding_dim=diffusion_step_embedding_dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    num_layers=num_layers,
                    add_upsampler=not is_last_block
                )
            )

            in_channels = out_channels

        self.conv_out = nn.Sequential(
            TrajectoryConv1d(in_channels, in_channels, kernel_size=5, num_groups=num_groups),
            nn.Conv1d(in_channels, transition_dim, kernel_size=1)
        )

    def forward(
        self,
        samples: torch.FloatTensor,
        diffusion_steps: torch.LongTensor
    ) -> TrajectoryUNetOutput:
        batch_size, horizon, transition_dim = samples.shape

        if horizon % 2 ** (len(self.downsample_blocks) - 1):
            raise ValueError(
                "Horizon must be divisible by the number of down/up-sampling processes.\n"
                f"Found {len(self.downsample_blocks) - 1} down/up-sampling processes, {horizon} long horizon."
            )

        # Check diffusion steps having batch dimension
        if len(diffusion_steps.shape) == 0:
            diffusion_steps = diffusion_steps.repeat(len(samples))
        elif len(diffusion_steps.shape) > 1:
            raise ValueError("Diffusion steps must be a single number or a batch of numbers")

        # Get current diffusion step embeddings
        diffusion_step_embeddings = self.diffusion_step_embedding(diffusion_steps)

        # Pre-process
        samples = samples.transpose(-2, -1)
        samples = self.conv_in(samples)

        # Down-sample
        skip_connection_samples_list = []

        for downsample_block in self.downsample_blocks:
            outputs = downsample_block(samples, diffusion_step_embeddings)

            samples = outputs.samples
            skip_connection_samples = outputs.skip_connection_samples

            skip_connection_samples_list.append(skip_connection_samples)

        # Mid
        samples = self.mid_block(samples, diffusion_step_embeddings).samples

        # Up-sample
        for upsample_block in self.upsample_blocks:
            outputs = upsample_block(samples, diffusion_step_embeddings, skip_connection_samples_list.pop())

            samples = outputs.samples

        # Post-process
        samples = self.conv_out(samples)
        samples = samples.transpose(-2, -1)

        return TrajectoryUNetOutput(samples=samples)
