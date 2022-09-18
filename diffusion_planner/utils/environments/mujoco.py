import numpy as np

from .gym import GymWrapperForRecording


class HopperWrapperForRecording(GymWrapperForRecording):
    def _set_state(self, observation: np.ndarray):
        qpos_dim, qvel_dim = self.env.model.nq, self.env.model.nv
        if observation.shape[-1] != qpos_dim + qvel_dim:
            qpos = np.concatenate([[0], observation[:qpos_dim - 1]])
        else:
            qpos = observation[:qpos_dim]
        qvel = observation[-qvel_dim:]

        self.env.set_state(qpos, qvel)
