python lal/real_robot_exp/pref_learning.py \
--env=widowx \
--data-dir=/scr/eisukehirota/correct-language-preference-learning/language-preference-learning/data/avoid_danger_user_study \
--model-dir=/scr/eisukehirota/correct-language-preference-learning/language-preference-learning/exp/widowx-img-obs_20240524_083517_lr_0.001_schedule_False \
--use-bert-encoder \
--lang-model=t5-small \
--use-img-obs \
--traj-encoder=cnn \
--method=active \
--lr=3e-3 \
--reward=3 \
--lang=3

# srun --nodes=1 --ntasks-per-node=1 --cpus-per-task=4 --mem-per-cpu=8G --gres=shard:1 --time=02:00:00 bash ./scripts/user_study_pref_learning_active.sh 