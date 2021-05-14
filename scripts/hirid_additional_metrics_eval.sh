#!/bin/bash

#bsub -o log_%J -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
#--dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
#--model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/dgpvae/models/hirid/comp/base/dim_8/len_25/scaled/n_scales_4/210126_n_1 \
#--assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_2.npy \
#--data_type_dci hirid --save_score --eval_type sap

for metric in modularity sap; do
  # dgp-vae
  for n in {1..10}; do
    bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/dgpvae/models/hirid/comp/base/dim_8/len_25/scaled/n_scales_4/hirid_"$metric"_"$n" \
    -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
    --dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
    --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/dgpvae/models/hirid/comp/base/dim_8/len_25/scaled/n_scales_4/210126_n_"$n" \
    --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_2.npy \
    --data_type_dci hirid --save_score --eval_type "$metric"
  done

  # adagvae
  for i in 210122_n_1_noband_1,1 210122_n_2_noband_1,2 210123_n_3_noband_1,3 210122_n_4_noband_1,4 210122_n_5_noband_1,5 210122_n_6_noband_1,6 210122_n_7_noband_1,7 210122_n_8_noband_1,8 210122_n_9_noband_1,9 210122_n_10_noband_1,10; do
    IFS=',' read model_name n <<< "${i}"
    bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/dgpvae/models/hirid/comp/ada/dim_8/hirid_"$metric"_"$n" \
    -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
    --dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/hirid_std.npz \
    --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/dgpvae/models/hirid/comp/ada/dim_8/"$model_name" \
    --assign_mat_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/hirid/assign_mats/assign_mat_2.npy \
    --data_type_dci hirid --save_score --eval_type "$metric"
  done
done