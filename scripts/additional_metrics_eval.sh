#!/bin/bash

bsub -o log_%J -g /gpvae_eval -R "rusage[mem=60000]" python eval_dci.py \
--dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range4.npz \
--model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/gp_full4/base/len_5/same/201130_n1 \
--data_type_dci dsprites --save_score --eval_type mig

# dSprites

# NORB

# cars3d

# shapes3d

# hirid