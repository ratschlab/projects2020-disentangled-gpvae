#!/bin/bash

#bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/gp_full4/log_%J -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
#--dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range4.npz \
#--model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/gp_full4/base/len_5/same/201130_n_1 \
#--data_type_dci dsprites --save_score --eval_type sap

#for metric in mig modularity sap;do
#  # dSprites
#  for i in 201130_n_1,1 201201_n_2,2 201201_n_3,3 201201_n_4,4 201201_n_5,5 210105_n_6,6 210106_n_7,7 210106_n_8,8 210106_n_9,9 210106_n_10,10; do
#    IFS=',' read model_name n <<< "${i}"
#    bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/gp_full4/dsprites_"$metric"_n_"$n" \
#    -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
#    --dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/dsprites/factors_dsprites_gp_full_range4.npz \
#    --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/gp_full4/base/len_5/same/"$model_name" \
#    --data_type_dci dsprites --save_score --eval_type "$metric"
#  done
#
#  # NORB
#  for i in 201224_n_1,1 201224_n_2,2 210127_n_21,3 201224_n_4,4 201225_n_5,5 201225_n_6,6 201225_n_7,7 201225_n_8,8 201225_n_9,9 210127_n_23,10; do
#    IFS=',' read model_name n <<< "${i}"
#    bsub -o log_norb_"$metric"_n_"$n" \
#    -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
#    --dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/norb/factors_norb_full1.npz \
#    --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/norb_full1/base/len_5/same/final_10/"$model_name" \
#    --data_type_dci smallnorb --save_score --eval_type "$metric"
#  done
# /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/norb_full1/norb_"$metric"_n_"$n"
#
#  # cars3d
  for i in 201229_n_1,1 201230_n_2,2 201230_n_3,3 201231_n_4,4 210130_n_22,5 201231_n_6,6 210101_n_7,7 210101_n_8,8 210101_n_9,9 210130_n_23,10; do
    IFS=',' read model_name n <<< "${i}"
    bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/cars_part1/cars_modularity_n_"$n"_2 \
    -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
    --dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_cars_part1.npz \
    --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/cars_part1/base/len_5/same/"$model_name" \
    --data_type_dci cars3d --save_score --eval_type modularity
  done
#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/cars_part1/cars_"$metric"_n_10 \
#  -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
#  --dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/cars3d/factors_cars_part1.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/cars_part1/base/len_5/same/final_10/210130_n_23 \
#  --data_type_dci cars3d --save_score --eval_type "$metric"
#
#  # shapes3d
#  for i in 210102_n_1,1 210102_n_2,2 210102_n_3,3 210102_n_4,4 210102_n_5,5 210102_n_6,6 210102_n_7,7 210102_n_8,8 210102_n_9,9 210103_n_10,10; do
#    IFS=',' read model_name n <<< "${i}"
#    bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/shapes_part2/shapes_"$metric"_n_"$n" \
#    -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
#    --dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_part2.npz \
#    --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/shapes_part2/base/len_5/same/"$model_name" \
#    --data_type_dci shapes3d --save_score --eval_type "$metric"
#  done
#  # hirid

#  bsub -o /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/shapes_part2/shapes_"$metric"_n_10 \
#  -g /gpvae_eval -R "rusage[mem=20000]" python eval_dci.py \
#  --dci_seed 0 --c_path /cluster/work/grlab/projects/projects2020_disentangled_gpvae/data/shapes3d/factors_shapes_part2.npz \
#  --model_name /cluster/work/grlab/projects/projects2020_disentangled_gpvae/models/shapes_part2/base/len_5/same/210103_n_10 \
#  --data_type_dci shapes3d --save_score --eval_type "$metric"
#done



