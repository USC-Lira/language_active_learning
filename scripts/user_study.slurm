#!/bin/bash
#SBATCH --job-name=user_study_active
#SBATCH --output=user_study_active.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=shard:1
#SBATCH --time=02:00:00

python lal/real_robot_exp/pref_learning.py \
--env=widowx \
--data-dir=/scr/eisukehirota/correct-language-preference-learning/language-preference-learning/data/avoid_danger_user_study \
--model-dir=/scr/eisukehirota/correct-language-preference-learning/language-preference-learning/exp/widowx-img-obs_20240524_083517_lr_0.001_schedule_False \
--use-bert-encoder \
--lang-model=t5-small \
--use-img-obs \
--traj-encoder=cnn \
--method=active \
--lr=3e-3