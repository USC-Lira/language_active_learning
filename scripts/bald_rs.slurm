#!/bin/bash
#SBATCH --job-name=rs_bald_3
#SBATCH --output=rs_bald_3.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=01:00:00

# Run your script
python lal/pref_learning/bald.py \
--env=robosuite \
--data-dir=/scr/eisukehirota/robosuite-benchmark/dataset/augmented-split-444-trajs \
--model-dir=exp/robosuite-img-obs-t5-small_20240508_211847_lr_0.001_schedule_False \
--true-reward-dir=lal/pref_learning/true_rewards_rs/2 \
--traj-encoder=cnn \
--lang-model=t5-small --use-bert-encoder \
--seed=1234 \
--lr=5e-3 \
--weight-decay=0.5 \
--num-iterations=20 \
--use-softmax \
--lang-temp=1.0 --use-constant-temp \
--use-img-obs --use-other-feedback --num-other-feedback=20 \
--num-models=50 \
--active=5 --lang=3 \
--dupe-traj=-1 \