import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch


def pad(array: np.ndarray, axis: int, length: int, value: int):
    padding_shape = list(array.shape)
    padding_shape[axis] = length
    padding = np.full(padding_shape, value, dtype=array.dtype)

    return np.concatenate([array, padding], axis=axis)


def to_tensor(data: np.ndarray):
    if data.dtype in {np.int8, np.int16, np.int32, np.int64, np.short, np.int, np.long, np.bool}:
        return torch.as_tensor(data, dtype=torch.long)
    return torch.as_tensor(data, dtype=torch.float)


@dataclass
class StaticCollatorWithPadding:
    pad_to: int

    def __post_init__(self):
        if self.pad_to < 2:
            raise ValueError("At least length two is needed for collation.")

    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        dict_of_lists = defaultdict(list)

        for idx, data in enumerate(batch):
            length = len(data["observations"])

            if self.pad_to < length:
                for name, _data in data.items():
                    dict_of_lists[name].append(_data[:self.pad_to])
            else:
                len_to_pad = self.pad_to - length
                if not len_to_pad:
                    continue

                for name, _data in data.items():
                    dict_of_lists[name].append(
                        pad(_data, axis=0, length=len_to_pad, value=name == "masks")
                    )

        return {
            name: to_tensor(np.stack(data, axis=0))
            for name, data in dict_of_lists.items()
        }


@dataclass
class DynamicCollatorWithPadding:
    pad_to_multiple_of: int

    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        dict_of_lists = defaultdict(list)

        max_length = max(len(data["observations"]) for data in batch)
        pad_to = math.ceil(max_length / self.pad_to_multiple_of) * self.pad_to_multiple_of

        for idx, data in enumerate(batch):
            len_to_pad = pad_to - len(data["observations"])
            if not len_to_pad:
                continue

            for name, _data in data.items():
                padded = pad(_data, axis=0, length=len_to_pad, value=name == "masks")
                dict_of_lists[name].append(
                    padded
                )

        return {
            name: to_tensor(np.stack(data, axis=0))
            for name, data in dict_of_lists.items()
        }
