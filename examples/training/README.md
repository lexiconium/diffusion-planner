## Training example

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lexiconium/diffusion-planner/blob/main/examples/training/run_diffusion_training.ipynb)

## CLI

```shell
python run_diffusion_training.py \
--path_or_url https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5 \
--pad_to_multiple_of 8 \
--trajectory_type dynamic \
--min_horizon 256 \
--block_out_channels 32 64 128 256 \
--num_layers_per_block 2 \
--num_groups 8 \
--output_dir outputs \
--do_train True \
--per_device_train_batch_size 512 \
--learning_rate 5e-5 \
--weight_decay 1e-2 \
--num_train_epochs 1 \
--warmup_ratio 0.3 \
--save_total_limit 5 \
--seed 42
```
