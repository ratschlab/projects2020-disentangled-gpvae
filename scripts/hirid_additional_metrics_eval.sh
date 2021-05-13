#!/bin/bash

bsub -o log_%J -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
--dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
--model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/dgpvae/models/hirid/comp/base/dim_8/len_25/scaled/n_scales_4/210126_n_1 \
--assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_2.npy \
--data_type_dci hirid --save_score --eval_type sap