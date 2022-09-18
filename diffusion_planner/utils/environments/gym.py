from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Tuple

import gym
import numpy as np
from gym import error, logger
from gym.core import ActType, ObsType
from gym.wrappers.monitoring.video_recorder import VideoRecorder


class NotAffectingEnvVideoRecorder(VideoRecorder):
    def close(self):
        """
        Same as VideoRecorder.close() except not closing the gym environment.
        """

        if not self.enabled or self._closed:
            return

        # Close the encoder
        if len(self.recorded_frames) > 0:
            try:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
            except ImportError:
                raise error.DependencyNotInstalled(
                    "MoviePy is not installed, run `pip install moviepy`"
                )

            logger.debug(f"Closing video encoder: path={self.path}")
            clip = ImageSequenceClip(self.recorded_frames, fps=self.frames_per_sec)
            clip.write_videofile(self.path)
        else:
            # No frames captured. Set metadata.
            if self.metadata is None:
                self.metadata = {}
            self.metadata["empty"] = True

        self.write_metadata()

        # Stop tracking this for autoclose
        self._closed = True


class OfflineRLUtilsForRecordingMixin(ABC):
    def step_with_observation(self, observation: np.ndarray):
        self._set_state(observation)

        if self._recorder is not None:
            self._recorder.capture_frame()

    def reset_with_observation(self, observation: np.ndarray):
        self.env.reset()
        self.step_with_observation(observation)

    @abstractmethod
    def _set_state(self, observation: np.ndarray):
        raise NotImplementedError


class GymWrapperForRecording(gym.Wrapper, OfflineRLUtilsForRecordingMixin, ABC):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self._recorder = None

    @contextmanager
    def record(self, path: str):
        self._recorder = NotAffectingEnvVideoRecorder(self.env, path=path)

        yield

        self._recorder.close()
        self._recorder = None

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        step_outputs = super().step(action)

        if self._recorder is not None:
            self._recorder.capture_frame()

        return step_outputs

    def reset(self, **kwargs) -> Tuple[ObsType, dict]:
        reset_outputs = super().reset(**kwargs)

        if self._recorder is not None:
            self._recorder.capture_frame()

        return reset_outputs

    def close(self):
        if self._recorder is not None:
            self._recorder.close()

        return super().close()


class ClassicControlWrapperForRecording(GymWrapperForRecording):
    def _set_state(self, observation: np.ndarray):
        self.env.state = observation
