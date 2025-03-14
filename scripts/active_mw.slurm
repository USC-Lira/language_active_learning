#!/bin/bash
#SBATCH --job-name=mw_1_3_3
#SBATCH --output=mw_1_3_3.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00

seeds=(0 1 2)
rewards=(0 1 2)

for reward in "${rewards[@]}"; do
    for seed in "${seeds[@]}"; do
        python lal/pref_learning/active_sim_pb.py \
        --env=metaworld \
        --method=lang \
        --data-dir=/scr/eisukehirota/language_active_learning/data/metaworld_original \
        --model-dir=exp/testing-mw-t5-base_20250312_235132_lr_0.001_schedule_False \
        --true-reward-dir=lal/pref_learning/true_rewards_mw/$reward \
        --traj-encoder=mlp \
        --lang-model=t5-base --use-bert-encoder \
        --seed=$seed \
        --use-softmax \
        --active=1 --reward=3 --lang=3 \
        --dupe-traj=-1
    done
done