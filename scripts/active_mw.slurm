#!/bin/bash
#SBATCH --job-name=mw_1_3_1
#SBATCH --output=mw_1_3_1.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00

seeds=(2 3 4 5 6)

for seed in "${seeds[@]}"; do
    python lal/pref_learning/active_sim_pb.py \
    --env=metaworld \
    --method=lang \
    --data-dir=/scr/eisukehirota/language_active_learning/data/metaworld_generated \
    --data-dir-extra=/scr/eisukehirota/language_active_learning/data/metaworld_add \
    --model-dir=exp/mw-t5-small_20240601_084347_lr_0.001_schedule_False \
    --true-reward-dir=lal/pref_learning/true_rewards_mw/$seed \
    --traj-encoder=mlp \
    --lang-model=t5-small --use-bert-encoder \
    --seed=1234 \
    --use-softmax \
    --use-img-obs \
    --active=1 --reward=3 --lang=1 \
    --dupe-traj=-1
done