#!/bin/bash
#SBATCH --job-name=4_4_1
#SBATCH --output=4_4_1.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00

# Run your script
python lal/pref_learning/active_sim_pb.py \
--method=lang \
--data-dir=data/data_pref_learning \
--model-dir=exp/robosuite-img-obs-t5-small_20240508_211847_lr_0.001_schedule_False \
--true-reward-dir=lal/pref_learning/true_rewards/1 \
--traj-encoder=cnn \
--lang-model=t5-small --use-bert-encoder \
--seed=0 \
--use-softmax \
--use-img-obs \
--active=4 --reward=4 --lang=1 \