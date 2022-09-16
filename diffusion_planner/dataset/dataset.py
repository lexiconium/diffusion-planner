import os
from collections import namedtuple
from dataclasses import dataclass, field
from typing import List

import datasets
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

DATA_FIELD_NAMES = ("observations", "actions", "next_observations", "rewards", "terminals", "timeouts")

DataField = namedtuple("DataField", DATA_FIELD_NAMES)
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

        # search fields for only the first depth, i.e. abandon nested data
        with h5py.File(self.path_or_url, "r") as f:
            data = {
                k: v[:] for k, v in f.items()
                if isinstance(v, h5py.Dataset) and k in DATA_FIELD_NAMES
            }

        if len(data.keys()) != len(DATA_FIELD_NAMES):
            # there could be some datasets that doesn't contain next_observations
            if "observations" in data and "next_observations" not in data:
                next_observations = data["observations"][1:].copy()
                data = {k: v[:-1] for k, v in data.items()}
                data["next_observations"] = next_observations

        if len(data.keys()) != len(DATA_FIELD_NAMES):
            raise ValueError(f"Data field {set(DATA_FIELD_NAMES) - set(data.keys())} missing.")

        # group by each step
        data = DataField(**data)
        data = [DataField(*_data) for _data in zip(*data)]

        # get episode begin indices
        episode_boundaries = [0]
        for idx, _data in enumerate(data):
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

        self._data = data

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, idx: int):
        begin, end = self._trajectories[idx]
        trajectory = self._data[begin:end]
        masks = np.zeros(end - begin)

        data = [np.stack(_data, axis=0) for _data in zip(*trajectory)] + [masks]

        return {name: _data for name, _data in zip(DATA_FIELD_NAMES + ("masks",), data)}

    @property
    def observation_dim(self):
        if not hasattr(self, "_observation_dim"):
            setattr(self, "_observation_dim", self._data[0].observations.shape[-1])
        return self._observation_dim

    @property
    def action_dim(self):
        if not hasattr(self, "_action_dim"):
            setattr(self, "_action_dim", self._data[0].actions.shape[-1])
        return self._action_dim
