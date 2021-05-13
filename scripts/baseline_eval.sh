#!/bin/bash

#bsub -o baselines/adagvae/log_sap_n_1 -g /disent_baseline -R "rusage[mem=30000]" \
#python baselines/eval_baseline.py --model adagvae --base_dir gp_full_4_2 --exp_name n_1 \
#--data dsprites_full --subset gp_full_4 --metric sap

for n in {1..10}; do
  for model in adagvae annealedvae betavae betatcvae factorvae dipvae_i dipvae_ii; do
    for metric in mig, modularity, sap; do
      for i in dsprites_full,gp_full_4,gp_full_4_2 smallnorb,norb_full1,norb_full1_1 cars3d,cars_part1,cars_part1_2 shapes3d,shapes_part2,shapes_part2_1; do
        IFS=',' read data subset base_dir <<< "${i}"
        bsub -o baselines/"$model"/log_"$metric"_n_"$n" -g /gpvae_eval -R "rusage[mem=30000]" \
        python baselines/eval_baseline.py --model "$model" --base_dir "$base_dir" --exp_name n_"$n" \
        --data "$data" --subset "$subset" --metric "$metric"
      done
    done
  done
done