#!/bin/bash
#SBATCH --job-name=rs_pbrl_lang
#SBATCH --output=rs_pbrl_lang.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00

# lrs=(5e-3)
lrs=(0.01)
# lrs=(0.05 0.1 0.5 1.0)
seeds=(0 1 2)
# seeds=(2)
# rewards=(2)
rewards=(0 1 2 3 4) # i think 0 1 2 3 are good, 4 is eh

for reward in "${rewards[@]}"; do
    for seed in "${seeds[@]}"; do
        for lr in "${lrs[@]}"; do
            python lal/pref_learning/pref_based_learning.py \
            --env=robosuite \
            --method=lang \
            --data-dir=/scr/eisukehirota/robosuite-benchmark/dataset/dense \
            --model-dir=exp/testing-rs-t5-base-both-norm-loss_20250212_122145_lr_0.001_schedule_False \
            --true-reward-dir=lal/pref_learning/true_rewards_rs/$reward \
            --traj-encoder=mlp \
            --lang-model=t5-base --use-bert-encoder \
            --seed=$seed \
            --lr=$lr \
            --weight-decay=0.2 \
            --num-iterations=30 \
            --use-softmax \
            --lang-temp=1.0 --use-constant-temp \
            --use-other-feedback --num-other-feedback=20 \
            --dupe-traj=-1 \
            --max-feature-norm=100.0 \
            --min-feature-norm=0.0
        done
    done
done

# datasets consist of: dense/, general/, early/, skewed/, skewed_new/
# --use-img-obs
# --traj-encoder=cnn