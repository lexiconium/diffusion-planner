import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import List

import datasets
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

_DATA_FIELD = ("observations", "actions", "next_observations", "rewards", "terminals", "timeouts")

DataField = namedtuple("DataField", _DATA_FIELD)
Trajectory = namedtuple("Trajectory", ["begin", "end"])


@dataclass
class OfflineRLDataset(Dataset):
    path_or_url: str
    horizon: int = field(default=64)
    minimum: int = field(default=16)

    _data: List[DataField] = field(init=False)
    _trajectories: List[Trajectory] = field(init=False)

    def __post_init__(self):
        # download in case not found in local path
        if not os.path.isfile(self.path_or_url):
            self.path_or_url = datasets.DownloadManager().download_and_extract(self.path_or_url)

        def loop_nested(data):
            if isinstance(data, h5py.Dataset):
                return data[:]
            if isinstance(data, h5py.Group):
                return {k: loop_nested(v) for k, v in data.items()}
            raise ValueError(f"Unexpected data type {type(data)} encountered.")

        # read file
        with h5py.File(self.path_or_url, "r") as f:
            data = {k: loop_nested(v) for k, v in f.items() if k in _DATA_FIELD}

        if len(data.keys()) != len(_DATA_FIELD):
            raise ValueError(f"Data field {set(_DATA_FIELD) - set(data.keys())} missing.")

        # group by each step
        data = DataField(**data)
        data = [DataField(*_data) for _data in zip(*data)]

        self._data = data

        # get episode begin indices
        episode_boundaries = [0]
        for idx, _data in enumerate(self._data):
            if _data.terminals or _data.timeouts:
                episode_boundaries.append(idx + 1)

        # set trajectories
        self._trajectories = []
        for episode_begin, episode_end in zip(tqdm(episode_boundaries[:-1]), episode_boundaries[1:]):
            if episode_end - episode_begin < self.minimum:
                continue

            for begin in range(episode_begin, episode_end - self.minimum):
                self._trajectories.append(
                    Trajectory(
                        begin=begin,
                        end=begin + np.random.randint(self.minimum, self.horizon + 1)
                    )
                )

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, idx: int):
        begin, end = self._trajectories[idx]
        trajectory = self._data[begin:end]
        mask = np.zeros(end - begin)

        return DataField(*[
            np.stack(field_data, axis=0)
            for field_data in zip(*trajectory)
        ]), mask
