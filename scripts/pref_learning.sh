python lal/pref_learning/pref_based_learning.py \
--method=lang \
--data-dir=../language-preference-learning/data/data_pref_learning \
--model-dir=exp/robosuite-img-obs-t5-small_20240601_192800_lr_0.001_schedule_False \
--true-reward-dir=lal/pref_learning/true_rewards/2 \
--traj-encoder=cnn \
--lang-model=t5-small --use-bert-encoder \
--seed=1234 \
--lr=5e-3 \
--weight-decay=0.1 \
--num-iterations=1 \
--use-softmax \
--lang-temp=1.0 --use-constant-temp \
--use-img-obs --use-other-feedback --num-other-feedback=10 \