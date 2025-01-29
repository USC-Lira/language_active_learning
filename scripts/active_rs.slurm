#!/bin/bash
#SBATCH --job-name=rs_1_3_3
#SBATCH --output=rs_1_3_3.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00

seeds=(0 1 2 3 4)
# seeds=(3 4)

for seed in "${seeds[@]}"; do
    python lal/pref_learning/active_sim_pb.py \
    --env=robosuite \
    --method=lang \
    --data-dir=/scr/eisukehirota/robosuite-benchmark/dataset/augmented-split-444-trajs \
    --model-dir=exp/robosuite-img-obs-t5-small_20240508_211847_lr_0.001_schedule_False \
    --true-reward-dir=lal/pref_learning/true_rewards_rs/$seed \
    --traj-encoder=cnn \
    --lang-model=t5-small --use-bert-encoder \
    --seed=1234 \
    --use-softmax \
    --use-img-obs \
    --active=1 --reward=3 --lang=3 \
    --dupe-traj=-1
done