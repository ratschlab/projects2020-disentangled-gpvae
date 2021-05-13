#!/bin/bash

bsub -o log_%J -g /disent_baseline -R "rusage[mem=60000]" \
python baselines/eval_baseline.py --model adagvae --base_dir gp_full_4_2 --exp_name n_1 \
--data dsprites --subset gp_full_4 --metric mig