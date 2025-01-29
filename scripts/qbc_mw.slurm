#!/bin/bash
#SBATCH --job-name=mw_qbc
#SBATCH --output=mw_qbc.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00

# lrs=(0.05 0.1 0.5 1.0)
lrs=(1.0)
seeds=(2 3 4 5 6)

for seed in "${seeds[@]}"; do
    for lr in "${lrs[@]}"; do
        python lal/pref_learning/qbc.py \
        --env=metaworld \
        --data-dir=/scr/eisukehirota/language_active_learning/data/metaworld_generated \
        --data-dir-extra=/scr/eisukehirota/language_active_learning/data/metaworld_add \
        --model-dir=exp/mw-t5-small_20240601_084347_lr_0.001_schedule_False \
        --true-reward-dir=lal/pref_learning/true_rewards_mw/$seed \
        --traj-encoder=mlp \
        --lang-model=t5-small --use-bert-encoder \
        --seed=1234 \
        --lr=$lr \
        --weight-decay=0 \
        --num-iterations=1 \
        --use-softmax \
        --lang-temp=1.0 --use-constant-temp \
        --use-img-obs --use-other-feedback --num-other-feedback=20 \
        --num-models=50 \
        --dupe-traj=-1
    done
done