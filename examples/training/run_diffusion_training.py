from dataclasses import dataclass, field

from diffusers.schedulers import DDPMScheduler
from transformers import HfArgumentParser, Trainer, TrainingArguments

from diffusion_planner.diffusers import TemporalUnetDiffuserForDDPM
from diffusion_planner.models import TemporalUnet
from diffusion_planner.utils.data import DatasetForD4RL, DynamicCollatorWithPadding


@dataclass
class DataArguments:
    path_or_url: str = field(metadata={"help": ""})
    pad_to_multiple_of: int = field(
        default=8, metadata={
            "help": "A constraint for dynamic padding strategy to avoid down-up sampling size mismatch in unet."
        }
    )


@dataclass
class ModelArguments:
    block_out_channels: int = field(default=(32, 64, 128, 256), metadata={"help": ""})
    num_layers_per_block: int = field(default=2, metadata={"help": ""})
    norm_eps: float = field(default=1e-5, metadata={"help": ""})
    num_groups: int = field(default=8, metadata={"help": ""})
    dropout: float = field(default=0.0, metadata={"help": ""})


def main():
    parser = HfArgumentParser([DataArguments, ModelArguments, TrainingArguments])
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    dataset = DatasetForD4RL(
        data_args.path_or_url,
        horizon=800,
        minimum=800
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
        dropout=ModelArguments.dropout
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
