#!/bin/bash
#SBATCH --job-name=pbrl
#SBATCH --output=pbrl.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:5
#SBATCH --time=01:00:00

# Run your script
python lal/pref_learning/pref_based_learning.py \
--env=metaworld \
--method=lang \
--data-dir=data/data_pref_learning \
--model-dir=exp/robosuite-img-obs-t5-small_20240508_211847_lr_0.001_schedule_False \
--true-reward-dir=lal/pref_learning/true_rewards_rs/3 \
--traj-encoder=cnn \
--lang-model=t5-small --use-bert-encoder \
--seed=1234 \
--lr=8e-3 \
--weight-decay=0.1 \
--num-iterations=1 \
--use-softmax \
--lang-temp=1.0 --use-constant-temp \
--use-img-obs --use-other-feedback --num-other-feedback=20 \