import math
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


def pad(array: np.ndarray, axis: int, length: int, value: int):
    padding_shape = list(array.shape)
    padding_shape[axis] = length
    padding = np.full(padding_shape, value, dtype=array.dtype)

    return np.concatenate([array, padding], axis=axis, dtype=array.dtype)


def to_tensor(data: np.ndarray):
    if data.dtype == np.bool:
        return torch.as_tensor(data, dtype=torch.bool)
    if data.dtype in {np.int8, np.int16, np.int32, np.int64, np.short, np.int, np.long}:
        return torch.as_tensor(data, dtype=torch.long)
    return torch.as_tensor(data, dtype=torch.float)


class DataCollator(ABC):
    @abstractmethod
    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        raise NotImplementedError("Collator must be defined.")


@dataclass
class DynamicDataCollatorWithPadding(DataCollator):
    pad_to_multiple_of: int

    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        dict_of_lists = defaultdict(list)

        max_trajectory_len = max(len(data["observations"]) for data in batch)
        pad_to = math.ceil(max_trajectory_len / self.pad_to_multiple_of) * self.pad_to_multiple_of

        for idx, data in enumerate(batch):
            trajectory_len = len(data["observations"])
            masks = np.zeros(trajectory_len, dtype=bool)

            # Fix initial state
            masks[0] = 1

            len_to_pad = pad_to - trajectory_len

            for name, field_data in data.items():
                dict_of_lists[name].append(pad(field_data, axis=0, length=len_to_pad, value=0))

            dict_of_lists["noise_masks"].append(pad(masks, axis=0, length=len_to_pad, value=1))

        return {
            name: to_tensor(np.stack(data, axis=0))
            for name, data in dict_of_lists.items()
        }


@dataclass
class StaticDataCollatorWithPadding(DataCollator):
    pad_to: int

    def __post_init__(self):
        if self.pad_to < 2:
            raise ValueError(f"Target length for padding must be greater than 1. Found {self.pad_to}")

    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        dict_of_lists = defaultdict(list)

        for idx, data in enumerate(batch):
            trajectory_len = len(data["observations"])
            masks = np.zeros(trajectory_len, dtype=bool)

            # Fix initial state
            masks[0] = 1

            len_to_pad = self.pad_to - trajectory_len

            if len_to_pad < 0:
                for name, field_data in data.items():
                    dict_of_lists[name].append(field_data[:self.pad_to])

                masks = masks[:self.pad_to]
            else:
                for name, field_data in data.items():
                    dict_of_lists[name].append(pad(field_data, axis=0, length=len_to_pad, value=0))

                masks = pad(masks, axis=0, length=len_to_pad, value=1)

            dict_of_lists["noise_masks"].append(masks)

        return {
            name: to_tensor(np.stack(data, axis=0))
            for name, data in dict_of_lists.items()
        }
