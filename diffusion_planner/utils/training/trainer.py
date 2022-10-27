from copy import deepcopy
from dataclasses import dataclass, field
from typing import Union

import torch
from accelerate import Accelerator
from diffusers.schedulers import DDIMScheduler, DDPMScheduler, PNDMScheduler
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .training_arguments import TrainingArguments


class EMAModel:
    def __init__(self, model: nn.Module):
        self.model = deepcopy(model)
        self.model.eval()

    def update(self, model: nn.Module, beta: float = 0.99):
        for params, _params in zip(self.model.parameters(), model.parameters()):
            params.data = beta * params.data + (1 - beta) * _params.data

    def unwrap(self) -> nn.Module:
        return self.model


@dataclass
class TrainerState:
    num_train_steps: int
    gradient_accumulation_steps: int
    ema_update_steps: int
    evaluation_steps: int
    save_steps: int

    step: int = field(init=False)

    def __post_init__(self):
        self.step = 0

    def update(self):
        self.step += 1

    @property
    def is_update_step(self) -> bool:
        return self.step % self.gradient_accumulation_steps == 0

    @property
    def is_ema_update_step(self) -> bool:
        return self.step % self.ema_update_steps == 0

    @property
    def is_eval_step(self) -> bool:
        return self.step % self.evaluation_steps == 0

    @property
    def is_save_step(self) -> bool:
        return self.step % self.save_steps == 0

    @property
    def is_train_end(self) -> bool:
        return self.step == self.num_train_steps


class Trainer:
    state: TrainerState

    def __init__(
        self,
        model: nn.Module,
        noise_scheduler: Union[DDPMScheduler, DDIMScheduler, PNDMScheduler],
        args: TrainingArguments,
        data_collator,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        optimizers: tuple[optim.Optimizer, optim.lr_scheduler.LambdaLR]
    ):
        self.model = model
        self.ema_model = EMAModel(model)

        self.noise_scheduler = noise_scheduler

        self.args = args

        self.data_collator = data_collator

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.optimizer, self.scheduler = optimizers

        self.accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)

        self.init_trainer_state()

    def init_trainer_state(self):
        args = self.args

        if args.num_train_steps is not None:
            num_train_steps = args.num_train_steps * args.gradient_accumulation_steps
        else:
            num_train_steps = (
                args.num_train_epochs
                * (len(self.train_dataset) // args.train_batch_size)
                * args.gradient_accumulation_steps
            )

        self.state = TrainerState(
            num_train_steps=num_train_steps,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            ema_update_steps=args.ema_update_steps,
            evaluation_steps=args.evaluation_steps,
            save_steps=args.save_steps
        )

    def train(self) -> nn.Module:
        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator
        )

        self.model, self.optimizer, self.scheduler, data_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.scheduler, data_loader
        )

        self.training_loop(data_loader)

        return self.ema_model.unwrap()

    def training_loop(self, data_loader: DataLoader):
        while not self.state.is_train_end:
            for data in tqdm(data_loader):
                with self.accelerator.accumulate(self.model):
                    trajectories = torch.cat([data["observations"], data["actions"]], dim=-1)
                    noise_masks = data["noise_masks"]

                    loss = self.compute_loss(trajectories, noise_masks)

                    self.accelerator.backward(loss)
                    self.state.update()

                    self.optimizer.step()
                    self.scheduler.step()

                    self.optimizer.zero_grad()

                    if self.state.is_ema_update_step:
                        self.ema_model.update(self.model, beta=self.args.ema_beta)

                    if self.state.is_eval_step:
                        self.evaluate()

                    if self.state.is_save_step:
                        self.save()

                    if self.state.is_train_end:
                        break

    def compute_loss(self, trajectories: torch.FloatTensor, noise_masks: torch.BoolTensor) -> torch.FloatTensor:
        noise = torch.randn_like(trajectories, device=trajectories.device)
        diffusion_steps = torch.randint(
            len(self.noise_scheduler),
            size=(trajectories.shape[0],),
            dtype=torch.long,
            device=trajectories.device
        )

        # Noise should not be applied where masked.
        noise[noise_masks] = 0

        noisy_trajectories = self.noise_scheduler.add_noise(trajectories, noise, diffusion_steps)

        noise_predictions = self.model(noisy_trajectories, diffusion_steps).samples

        loss = nn.functional.mse_loss(noise_predictions, noise)

        return loss

    def evaluate(self):
        pass

    def evaluation_loop(self):
        pass

    def evaluation_step(self):
        pass

    def save(self):
        pass
