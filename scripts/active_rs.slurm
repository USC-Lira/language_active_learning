#!/bin/bash
#SBATCH --job-name=rs_1_3_3
#SBATCH --output=rs_1_3_3.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00

seeds=(0 1 2)
rewards=(0 1 2 3 4)
# seeds=(0)
# rewards=(0)

for reward in "${rewards[@]}"; do
    for seed in "${seeds[@]}"; do
        python lal/pref_learning/active_sim_pb.py \
        --env=robosuite \
        --method=lang \
        --data-dir=/scr/eisukehirota/robosuite-benchmark/dataset/dense \
        --model-dir=exp/testing-rs-t5-base-both-norm-loss_20250212_122145_lr_0.001_schedule_False \
        --true-reward-dir=lal/pref_learning/true_rewards_rs/$reward \
        --traj-encoder=mlp \
        --lang-model=t5-base --use-bert-encoder \
        --seed=$seed \
        --use-softmax \
        --active=1 --reward=3 --lang=3 \
        --dupe-traj=-1
    done
done

        # --use-img-obs \