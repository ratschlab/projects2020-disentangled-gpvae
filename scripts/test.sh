#!/bin/bash

# Test script for job submission with scripts
SEED=$RANDOM
for dim in 8 16; do
  mkdir -p models/dsprites_dim_$dim
  bsub -g /gpvae_disent -o /models/dsprites_dim_"$dim" -R "rusage[mem=16000,ngpus_excl_p=1]" \
  python train.py --model_type gp-vae --data_type dsprites --exp_name dsprites_"$dim"_n$n \
  --basedir models/dsprites_dim_"$dim" --seed $SEED --banded_covar --latent_dim $dim \
  --encoder_sizes=32,256,256 --decoder_sizes=256,256,256 --window_size 3 --sigma 1 \
  --length_scale 2 --beta 1.0 --num_epochs 20
done
