import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


def get_add_shape(shape: Tuple[int, ...], *, len_to_add: int):
    if len(shape) == 1:
        return (len_to_add,)
    if len(shape) == 2:
        return (len_to_add, shape[1])
    raise ValueError("Data with three or more dimensions are not supported.")


def ndarray_to_tensor(data: np.ndarray):
    if data.dtype in {np.int8, np.int16, np.int32, np.int64, np.short, np.int, np.long, np.bool}:
        return torch.as_tensor(data, dtype=torch.long)
    return torch.as_tensor(data, dtype=torch.float)


@dataclass
class StaticCollator:
    collate_to: int

    def __post_init__(self):
        if self.collate_to < 2:
            raise ValueError("At least length two is needed for collation.")

    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        dict_of_lists = defaultdict(list)

        for idx, data in enumerate(batch):
            length = len(data["observations"])

            if self.collate_to < length:
                for name, _data in data.items():
                    dict_of_lists[name].append(_data[:self.collate_to])
            else:
                len_to_add = self.collate_to - length
                if not len_to_add:
                    continue

                for name, _data in data.items():
                    dict_of_lists[name].append(
                        np.concatenate(
                            [_data,
                             np.zeros(get_add_shape(_data.shape, len_to_add=len_to_add))
                             if name != "masks" else
                             np.ones(len_to_add)],
                            axis=0
                        )
                    )

        return {
            name: ndarray_to_tensor(np.stack(data, axis=0))
            for name, data in dict_of_lists.items()
        }


@dataclass
class DynamicCollator:
    collate_to_multiple_of: int

    def __call__(self, batch: List[Dict[str, np.ndarray]]):
        dict_of_lists = defaultdict(list)

        max_length = max(len(data["observations"]) for data in batch)
        collate_to = math.ceil(max_length / self.collate_to_multiple_of) * self.collate_to_multiple_of

        for idx, data in enumerate(batch):
            len_to_add = collate_to - len(data["observations"])
            if not len_to_add:
                continue

            for name, _data in data.items():
                dict_of_lists[name].append(
                    np.concatenate(
                        [_data,
                         np.zeros(get_add_shape(_data.shape, len_to_add=len_to_add))
                         if name != "masks" else
                         np.ones(len_to_add)],
                        axis=0
                    )
                )

        return {
            name: ndarray_to_tensor(np.stack(data, axis=0))
            for name, data in dict_of_lists.items()
        }
