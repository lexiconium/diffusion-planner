from dataclasses import dataclass

import gym
import numpy as np
import torch
from diffusers.schedulers import DDPMScheduler, DDIMScheduler
from transformers import HfArgumentParser

from diffusion_planner import DiffusionPlanner
from diffusion_planner.models import TemporalUnet
from diffusion_planner.utils.environments import HopperWrapperForRecording

config_path = "/Users/mskim/diffusion-planner/outputs/model_config.json"
checkpoint = "/Users/mskim/diffusion-planner/outputs/pytorch_model.bin"


@dataclass
class PlannerArguments:
    ...


def main():
    parser = HfArgumentParser([])
    model_args = parser.parse_args_into_dataclasses()

    unet = TemporalUnet.from_config(config_path)
    unet.load_state_dict({
        name[5:]: param
        for name, param in torch.load(checkpoint, map_location="cpu").items()
    })

    planner = DiffusionPlanner(unet)

    scheduler = DDPMScheduler()
    # scheduler = DDIMScheduler()

    env = gym.make("Hopper-v3", render_mode="rgb_array")
    env = HopperWrapperForRecording(env)

    with env.record("video.mp4"):
        observation, _ = env.reset()
        observation = observation
        observation_dim = len(observation)

        sample: np.ndarray = planner(
            observation,
            scheduler=scheduler,
            horizon=512,
            num_inference_steps=200
        ).sample.squeeze(axis=0)

        for _sample in sample:
            observation, action = _sample[:observation_dim], _sample[observation_dim:]
            # env.step(action)

            env.step_with_observation(observation)

            print(observation[0])

    env.close()


if __name__ == "__main__":
    main()
