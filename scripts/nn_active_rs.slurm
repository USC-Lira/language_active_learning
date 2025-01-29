#!/bin/bash
#SBATCH --job-name=rs_nn_active_1_3_3
#SBATCH --output=rs_nn_active_1_3_3.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00

lrs=(5e-3)
seeds=(0 1 2 3 4)
# seeds=(3 4)
# seeds=(3 4)

for seed in "${seeds[@]}"; do
    for lr in "${lrs[@]}"; do
        python lal/pref_learning/nn_active.py \
        --env=robosuite \
        --method=lang \
        --data-dir=/scr/eisukehirota/robosuite-benchmark/dataset/augmented-split-444-trajs \
        --model-dir=exp/robosuite-img-obs-t5-small_20240508_211847_lr_0.001_schedule_False \
        --true-reward-dir=lal/pref_learning/true_rewards_rs/$seed \
        --traj-encoder=cnn \
        --lang-model=t5-small --use-bert-encoder \
        --seed=1234 \
        --lr=$lr \
        --weight-decay=0.2 \
        --num-iterations=20 \
        --use-softmax \
        --lang-temp=1.0 --use-constant-temp \
        --use-img-obs --use-other-feedback --num-other-feedback=20 \
        --active=1 --reward=3 --lang=3 \
        --dupe-traj=-1 \
        --max-feature-norm=100.0 \
        --min-feature-norm=0.0
    done
done