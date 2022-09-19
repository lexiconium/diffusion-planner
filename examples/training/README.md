## Training example

```shell
python run_diffusion_training.py \
--path_or_url https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5 \
--pad_to_multiple_of 8 \
--trajectory_type dynamic \
--min_horizon 256 \
--block_out_channels 32 64 128 256 \
--num_layers_per_block 8 \
--num_groups 8 \
--output_dir outputs \
--do_train True \
--per_device_train_batch_size 8 \
--learning_rate 2e-4 \
--weight_decay 1e-2 \
--num_train_epochs 1 \
--warmup_ratio 0.3 \
--save_total_limit 5 \
--seed 42
```
