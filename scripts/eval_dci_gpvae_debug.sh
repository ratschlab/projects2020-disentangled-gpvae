#!/bin/bash

# Evaluate DCI metric for gpvae models

for dim in 8 16 32 64 128; do
  for base_dir in models/dsprites_dim_"$dim"_debug3/*; do
    bsub -g /gpvae_norm -R "rusage[mem=16000]" python dsprites_dci.py --z_name factors_5000.npy \
    --model_name $base_dir
  done
done
