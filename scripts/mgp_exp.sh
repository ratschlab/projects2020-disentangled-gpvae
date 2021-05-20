#!/bin/bash

#bsub -o log_bb_%J -g /gpvae_disent \
#-R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#python run_experiment.py --model_type gp-vae --data_type cars3d --time_len 5 --testing --batch_size 32 \
#--data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/cars_part1.npz \
#--exp_name n_0 --basedir models/mgpvae/bb/test \
#--seed 0 --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init same \
#--decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
#--num_epochs 1 --kernel bb --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_cars_part1.npz \
#--save_score
#
#bsub -o log_fbm_%J -g /gpvae_disent \
#-R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
#python run_experiment.py --model_type gp-vae --data_type cars3d --time_len 5 --testing --batch_size 32 \
#--data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/cars_part1.npz \
#--exp_name n_0 --basedir models/mgpvae/bb/test \
#--seed 0 --banded_covar --latent_dim 64 --kernel_scales 64 --encoder_sizes=32,256,256 --len_init same \
#--decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
#--num_epochs 1 --kernel fbm --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_cars_part1.npz \
#--save_score

for n in {1..10}; do
  seed=$RANDOM
  for i in dsprites,dsprites_gp_full_range4 smallnorb,norb_full1 cars3d,cars_part1 shapes3d,shapes_part2; do
    IFS=',' read dataset data_name <<< "${i}"
    # Brownian bridge kernel
    bsub -o log_t_bb_n"$n"_"$dataset"_%J -g /gpvae_norm \
    -R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]" \
    python run_experiment.py --model_type dgp-vae --data_type "$dataset" --time_len 5 --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/"$dataset"/"$data_name".npz \
    --exp_name n_"$n"_t --basedir models/mgpvae/bb/"$dataset" \
    --seed "$seed" --banded_covar --latent_dim 64 --encoder_sizes=32,256,256 --len_init same \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 1 --kernel bb --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/"$dataset"/factors_"$data_name".npz \
    --save_score

    # Fractional brownian motion kernel
    bsub -o log_t_fbm_n"$n"_"$dataset"_%J -g /gpvae_norm \
    -R "rusage[mem=220000,ngpus_excl_p=1]" -R "select[gpu_model0==TeslaV100_SXM2_32GB]"\
    python run_experiment.py --model_type dgp-vae --data_type "$dataset" --time_len 5 --testing --batch_size 32 \
    --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/"$dataset"/"$data_name".npz \
    --exp_name n_"$n"_t --basedir models/mgpvae/fbm/"$dataset" \
    --seed "$seed" --banded_covar --latent_dim 64 --kernel_scales 64 --encoder_sizes=32,256,256 --len_init same \
    --decoder_sizes=256,256,256 --window_size 3 --sigma 1 --length_scale 2 --beta 1.0 \
    --num_epochs 1 --kernel fbm --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/"$dataset"/factors_"$data_name".npz \
    --save_score
  done
done