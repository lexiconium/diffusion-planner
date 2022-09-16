from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from .dataset import DataField


def get_add_shape(shape: Tuple[int, ...], *, len_to_add: int):
    if len(shape) == 1:
        return (len_to_add,)
    if len(shape) == 2:
        return (len_to_add, shape[1])
    raise ValueError("Data with three or more dimensions are not supported.")


@dataclass
class StaticCollator:
    collate_to: int

    def __post_init__(self):
        if self.collate_to < 2:
            raise ValueError("At least length two is needed for collation.")

    def __call__(self, batch: List[Tuple[DataField, np.ndarray]]):
        for idx, (data, masks) in enumerate(batch):
            if self.collate_to < len(masks):
                batch[idx] = (
                    DataField(*[_data[:self.collate_to] for _data in data]),
                    masks[:self.collate_to]
                )
            else:
                len_to_add = self.collate_to - len(data.observations)
                if not len_to_add:
                    continue

                batch[idx] = (
                    DataField(*[
                        np.concatenate(
                            [_data, np.zeros(get_add_shape(_data.shape, len_to_add=len_to_add))],
                            axis=0
                        ) for _data in data
                    ]),
                    np.concatenate([masks, np.ones(len_to_add)])
                )

        batch, masks = zip(*batch)

        batch = DataField(*[torch.as_tensor(np.stack(data, axis=0)) for data in zip(*batch)])
        masks = torch.as_tensor(np.stack(masks, axis=0))

        return batch, masks


class DynamicCollator:
    def __call__(self, batch: List[Tuple[DataField, np.ndarray]]):
        max_length = max(len(data.observations) for data, _ in batch)

        for idx, (data, masks) in enumerate(batch):
            len_to_add = max_length - len(data.observations)
            if not len_to_add:
                continue

            batch[idx] = (
                DataField(*[
                    np.concatenate(
                        [_data, np.zeros(get_add_shape(_data.shape, len_to_add=len_to_add))],
                        axis=0
                    ) for _data in data
                ]),
                np.concatenate([masks, np.ones(len_to_add)])
            )

        batch, masks = zip(*batch)

        batch = DataField(*[torch.as_tensor(np.stack(data, axis=0)) for data in zip(*batch)])
        masks = torch.as_tensor(np.stack(masks, axis=0))

        return batch, masks
