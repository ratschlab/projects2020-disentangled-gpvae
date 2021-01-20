#!/bin/bash

mkdir -p models/hirid/std/dim_8/len_5/same
mkdir -p models/hirid/no_std/dim_8/len_5/same
mkdir -p models/hirid/std/dim_16/len_5/same
mkdir -p models/hirid/no_std/dim_16/len_5/same
mkdir -p models/hirid/std/dim_32/len_5/same
mkdir -p models/hirid/no_std/dim_32/len_5/same

mkdir -p models/hirid/std/dim_8/len_25/same
mkdir -p models/hirid/no_std/dim_8/len_25/same
mkdir -p models/hirid/std/dim_16/len_25/same
mkdir -p models/hirid/no_std/dim_16/len_25/same
mkdir -p models/hirid/std/dim_32/len_25/same
mkdir -p models/hirid/no_std/dim_32/len_25/same

mkdir -p models/hirid/std/dim_8/len_50/same
mkdir -p models/hirid/no_std/dim_8/len_50/same
mkdir -p models/hirid/std/dim_16/len_50/same
mkdir -p models/hirid/no_std/dim_16/len_50/same
mkdir -p models/hirid/std/dim_32/len_50/same
mkdir -p models/hirid/no_std/dim_32/len_50/same

mkdir -p models/hirid/std/dim_8/len_5/scaled
mkdir -p models/hirid/no_std/dim_8/len_5/scaled
mkdir -p models/hirid/std/dim_16/len_5/scaled
mkdir -p models/hirid/no_std/dim_16/len_5/scaled
mkdir -p models/hirid/std/dim_32/len_5/scaled
mkdir -p models/hirid/no_std/dim_32/len_5/scaled

mkdir -p models/hirid/std/dim_8/len_25/scaled
mkdir -p models/hirid/no_std/dim_8/len_25/scaled
mkdir -p models/hirid/std/dim_16/len_25/scaled
mkdir -p models/hirid/no_std/dim_16/len_25/scaled
mkdir -p models/hirid/std/dim_32/len_25/scaled
mkdir -p models/hirid/no_std/dim_32/len_25/scaled

mkdir -p models/hirid/std/dim_8/len_50/scaled
mkdir -p models/hirid/no_std/dim_8/len_50/scaled
mkdir -p models/hirid/std/dim_16/len_50/scaled
mkdir -p models/hirid/no_std/dim_16/len_50/scaled
mkdir -p models/hirid/std/dim_32/len_50/scaled
mkdir -p models/hirid/no_std/dim_32/len_50/scaled

mkdir -p models/hirid/ada/std/dim_8
mkdir -p models/hirid/ada/std/dim_16
mkdir -p models/hirid/ada/std/dim_32
mkdir -p models/hirid/no_ada/std/dim_8
mkdir -p models/hirid/no_ada/std/dim_16
mkdir -p models/hirid/no_ada/std/dim_32

for n in {6..10}; do
  seed=$RANDOM
  for sens_eval_type in std; do
    for lat_dim in 8 16 32; do
      bsub -o models/hirid/ada/"$sens_eval_type"/dim_"$lat_dim"/log_%J -g /gpvae_norm -W 48:00 -R "rusage[mem=65000,ngpus_excl_p=1]" \
      -R "select[gpu_model0==GeForceGTX1080Ti]" python run_experiment.py --model_type ada-gp-vae \
      --data_type hirid --time_len 1 --testing --batch_size 64 --exp_name n_"$n" \
      --basedir models/hirid/ada/"$sens_eval_type"/dim_"$lat_dim" --len_init same \
      --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
      --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 --decoder_sizes=256,256 \
      --window_size 1 --sigma 1.005 --beta 1.0 --num_epochs 1 --kernel id \
      --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
      --visualize_score --save_score --eval_type dci --data_type_dci hirid --shuffle
      for len in 5,3,24 25,12,4 50,25,4; do
        IFS=',' read time_len window_size run_time <<< "${len}"
        bsub -o models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/same/log_%J \
        -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
        python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/same \
        --len_init same --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 \
        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 2.0 --beta 1.0 \
        --num_epochs 1 --kernel cauchy \
        --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
        --visualize --save --eval_type dci \
        --data_type_dci hirid --shuffle

        bsub -o models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/scaled/log_%J \
        -g /gpvae_disent -W "$run_time":00 -R "rusage[mem=65000,ngpus_excl_p=1]" -R "select[gpu_model0==GeForceGTX1080Ti]" \
        python run_experiment.py --model_type gp-vae --data_type hirid --time_len "$time_len" \
        --testing --batch_size 64 --exp_name n_"$n" --basedir models/hirid/"$sens_eval_type"/dim_"$lat_dim"/len_"$time_len"/scaled \
        --len_init scaled --data_dir /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
        --seed "$seed" --banded_covar --latent_dim "$lat_dim" --encoder_sizes=128,128 --kernel_scales "$lat_dim"\
        --decoder_sizes=256,256 --window_size "$window_size" --sigma 1.005 --length_scale 20.0 --beta 1.0 \
        --num_epochs 1 --kernel cauchy \
        --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_"$sens_eval_type".npz \
        --visualize_score --save_score --eval_type dci \
        --data_type_dci hirid --shuffle

      done
    done
  done
done