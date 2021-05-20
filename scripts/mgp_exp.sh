#!/bin/bash

bsub -o log_bb_%J -g /gpvae_disent \
-R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
python run_experiment.py --model_type gp-vae --data_type cars3d --time_len 5 --testing --batch_size 32 \
--data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/cars_part1.npz \
--exp_name n_0 --basedir models/mgpvae/bb \
--seed 0 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init same \
--decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
--num_epochs 1 --kernel bb --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_cars_part1.npz \
--save_score