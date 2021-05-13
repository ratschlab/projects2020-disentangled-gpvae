#!/bin/bash

for model in adagvae annealedvae betavae betatcvae factorvae dipvae_i dipvae_ii; do
  for metric in mig modularity sap; do
    for exp_name in gp_full_4_2 norb_full1_1 cars_part1_2 shapes_part2_1; do
      python lib/dci_aggregate.py --model "$model" --exp_name "$exp_name" --save --metric "$metric"
    done
  done
done