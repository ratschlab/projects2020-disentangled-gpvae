#!/bin/bash

bsub -o baselines/adagvae/log_sap_n_1 -g /disent_baseline -R "rusage[mem=10000]" \
python baselines/eval_baseline.py --model adagvae --base_dir gp_full_4_2 --exp_name n_1 \
--data dsprites_full --subset gp_full_4 --metric sap