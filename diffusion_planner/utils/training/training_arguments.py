from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainingArguments:
    num_train_epochs: Optional[int] = field(default=None)
    num_train_steps: Optional[int] = field(default=None)
    train_batch_size: int = field(default=8)
    eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=4)
    ema_beta: float = field(default=0.99)
    ema_update_steps: int = field(default=10)
    evaluation_steps: int = field(default=500)
    save_steps: Optional[int] = field(default=None)

    def __post_init__(self):
        if self.num_train_epochs is None and self.num_train_steps is None:
            raise ValueError("Either one of num_train_epochs or num_train_steps should be provided.")
