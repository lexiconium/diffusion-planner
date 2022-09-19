```shell
python run_diffusion_training.py \
--path_or_url https://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco_v2/hopper_expert-v2.hdf5 \
--output_dir outputs \
--do_train True \
--per_device_train_batch_size 8 \
--save_total_limit 10
```
