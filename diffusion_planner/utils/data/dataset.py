import os
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from typing import Dict, Tuple

import datasets
import h5py
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

Trajectory = namedtuple("Trajectory", ["begin", "end"])


class DatasetForOfflineRL(Dataset, ABC):
    path_or_url: str
    horizon: int
    minimum: int

    _fields: Tuple[str]

    def __post_init__(self):
        # download in case not found in local path
        if not os.path.isfile(self.path_or_url):
            self.path_or_url = datasets.DownloadManager().download_and_extract(self.path_or_url)

        self._data = self._load_data()

        if len(self._data.keys()) != len(self._fields):
            # there could be some datasets that doesn't contain next_observations
            if "observations" in self._data and "next_observations" not in self._data:
                next_observations = self._data["observations"][1:].copy()
                self._data = dict(
                    {k: v[:-1] for k, v in self._data.items()},
                    next_observations=next_observations
                )

        if len(self._data.keys()) != len(self._fields):
            raise ValueError(f"Data field {set(self._fields) - set(self._data.keys())} missing.")

        self._observation_dim = self._data["observations"].shape[-1]
        self._action_dim = self._data["actions"].shape[-1]

        # get episode begin indices
        episode_boundaries = [0]
        for idx, (terminals, timeouts) in enumerate(zip(self._data["terminals"], self._data["timeouts"])):
            if terminals or timeouts:
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
                        end=begin + np.random.randint(
                            self.minimum,
                            min(self.horizon, episode_end - begin) + 1
                        )
                    )
                )

    @abstractmethod
    def _load_data(self) -> Dict[str, np.ndarray]:
        """
        A data loading method to be implemented.
        Returned observations and actions must be a shape of (number of steps, {observation, action} dimension)
        """

        raise NotImplementedError

    def __len__(self):
        return len(self._trajectories)

    def __getitem__(self, idx: int):
        begin, end = self._trajectories[idx]

        return dict(
            {key: value[begin:end] for key, value in self._data.items()},
            masks=np.zeros(end - begin)
        )

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def action_dim(self):
        return self._action_dim


@dataclass
class DatasetForD4RL(DatasetForOfflineRL):
    path_or_url: str
    horizon: int
    minimum: int

    _fields: Tuple[str] = field(
        default=("observations", "actions", "next_observations", "rewards", "terminals", "timeouts"),
        init=False,
        repr=False
    )

    def _load_data(self) -> Dict[str, np.ndarray]:
        def broadcast_if_needed(data: np.ndarray):
            num_dim = len(data.shape)
            if num_dim == 1:
                return data[:, None]
            if num_dim == 2:
                return data
            raise ValueError("Observation and action must be a scalar or vector.")

        with h5py.File(self.path_or_url, "r") as f:
            data = {
                k: broadcast_if_needed(v[:]) if k in {"observations", "actions", "next_observations"} else v[:]
                for k, v in f.items()
                if isinstance(v, h5py.Dataset) and k in self._fields
            }

        return data
