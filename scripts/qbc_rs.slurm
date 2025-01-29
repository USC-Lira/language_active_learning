#!/bin/bash
#SBATCH --job-name=rs_qbc
#SBATCH --output=rs_qbc.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00

lrs=(5e-3)
seeds=(0 1 2 3 4)

for seed in "${seeds[@]}"; do
    for lr in "${lrs[@]}"; do
        python lal/pref_learning/qbc.py \
        --env=robosuite \
        --data-dir=data/robosuite_pref_learning \
        --model-dir=exp/robosuite-img-obs-t5-small_20240508_211847_lr_0.001_schedule_False \
        --true-reward-dir=lal/pref_learning/true_rewards_rs/$seed \
        --traj-encoder=cnn \
        --lang-model=t5-small --use-bert-encoder \
        --seed=1234 \
        --lr=$lr \
        --weight-decay=0.1 \
        --num-iterations=5 \
        --use-softmax \
        --lang-temp=1.0 --use-constant-temp \
        --use-img-obs --use-other-feedback --num-other-feedback=20 \
        --num-models=50
    done
done