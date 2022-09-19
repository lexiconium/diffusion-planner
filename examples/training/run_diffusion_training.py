from dataclasses import dataclass, field
from typing import Optional, Union

from diffusers.schedulers import DDPMScheduler
from transformers import HfArgumentParser, Trainer, TrainingArguments

from diffusion_planner.diffusers import TemporalUnetDiffuserForDDPM
from diffusion_planner.models import TemporalUnet
from diffusion_planner.utils.data import DatasetForD4RL, DynamicCollatorWithPadding, TrajectoryType


@dataclass
class DataArguments:
    path_or_url: str = field(metadata={"help": "Path or URL of the dataset."})
    pad_to_multiple_of: int = field(
        default=8, metadata={
            "help": "A constraint for dynamic padding strategy to avoid down-up sampling size mismatch in unet."
        }
    )
    trajectory_type: Union[TrajectoryType, str] = field(
        default="dynamic",
        metadata={
            "help": "Type of trajectory. If it's `fixed`, it forms trajectories of length `horizon`."
                    " If `dynamic`, It randomly samples a length between `max_horizon` and `min_horizon`."
        }
    )
    horizon: Optional[int] = field(
        default=None, metadata={"help": "Length of trajectory. Only used when `trajectory_type` is `fixed`."}
    )
    max_horizon: Optional[int] = field(
        default=None,
        metadata={
            "help": "An upper bound used for forming trajectories."
                    " This is only used when `trajectory_type` is dynamic."
                    " If None, episode horizon is used instead."
        }
    )
    min_horizon: int = field(
        default=256,
        metadata={
            "help": "A lower bound used for forming trajectories."
                    " This is only used when `trajectory_type` is dynamic."
        }
    )


@dataclass
class ModelArguments:
    block_out_channels: int = field(
        default=(32, 64, 128, 256), metadata={"help": "Unet block out channels."}
    )
    num_layers_per_block: int = field(default=2, metadata={"help": "Number of resnets per block."})
    norm_eps: float = field(
        default=1e-5,
        metadata={"help": "A value added to the denominator for numerical stability in group normalization."}
    )
    num_groups: int = field(
        default=8,
        metadata={
            "help": "Number of groups in group normalization."
                    " Block out channels must a multiple of this value."
        }
    )
    dropout: float = field(default=0.0, metadata={"help": "Dropout for resnet."})


def main():
    parser = HfArgumentParser([DataArguments, ModelArguments, TrainingArguments])
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    dataset = DatasetForD4RL(
        data_args.path_or_url,
        trajectory_type=data_args.trajectory_type,
        horizon=data_args.horizon,
        max_horizon=data_args.max_horizon,
        min_horizon=DataArguments.min_horizon
    )

    transition_dim = dataset.observation_dim + dataset.action_dim

    scheduler = DDPMScheduler()
    unet = TemporalUnet(
        in_channels=transition_dim,
        out_channels=transition_dim,
        block_out_channels=model_args.block_out_channels,
        num_layers_per_block=model_args.num_layers_per_block,
        norm_eps=model_args.norm_eps,
        num_groups=model_args.num_groups,
        dropout=model_args.dropout
    )

    model = TemporalUnetDiffuserForDDPM(unet=unet, scheduler=scheduler)

    trainer = Trainer(
        model,
        args=training_args,
        data_collator=DynamicCollatorWithPadding(
            pad_to_multiple_of=data_args.pad_to_multiple_of
        ),
        train_dataset=dataset
    )

    trainer.train()


if __name__ == "__main__":
    main()
