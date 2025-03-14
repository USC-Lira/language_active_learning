python lang_pref_learning/real_robot_exp/pref_learning.py \
--env=widowx \
--data-dir=data/avoid_danger_user_study \
--model-dir=exp/widowx-img-obs_20240524_083517_lr_0.001_schedule_False \
--use-bert-encoder \
--lang-model=t5-small \
--use-img-obs \
--lr=3e-3 \
--method=comp \
--traj-encoder=cnn \
# --real-robot # in zhaojing's script, this flag is supposed to be here and lr should be 1e-3 instead of 3e-3. but for some reason
# in language, lr is 3e-3. i'm turning the flag off just for testing purposes