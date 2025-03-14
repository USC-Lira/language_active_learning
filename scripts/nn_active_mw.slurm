#!/bin/bash
#SBATCH --job-name=mw_nn_active_1_3_3
#SBATCH --output=mw_nn_active_1_3_3.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00

# lrs=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
# lrs=(0.05 0.1 0.5 1.0)
lrs=(0.03) # best is 0.03 w/ wd=0.3
seeds=(0 1 2)
# seeds=(0)
rewards=(0 1 2)
# rewards=(0)

for reward in "${rewards[@]}"; do
    for seed in "${seeds[@]}"; do
        for lr in "${lrs[@]}"; do
            python lal/pref_learning/nn_active.py \
            --env=metaworld \
            --method=lang \
            --data-dir=/scr/eisukehirota/language_active_learning/data/metaworld_original \
            --model-dir=exp/testing-mw-t5-base_20250312_235132_lr_0.001_schedule_False \
            --true-reward-dir=lal/pref_learning/true_rewards_mw/$reward \
            --traj-encoder=mlp \
            --lang-model=t5-base --use-bert-encoder \
            --seed=$seed \
            --lr=$lr \
            --weight-decay=0.3 \
            --num-iterations=1 \
            --use-softmax \
            --lang-temp=1.0 --use-constant-temp \
            --use-other-feedback --num-other-feedback=20 \
            --active=1 --reward=3 --lang=3 \
            --dupe-traj=-1 \
            --max-feature-norm=3.0 \
            --min-feature-norm=0.0
        done
    done
done