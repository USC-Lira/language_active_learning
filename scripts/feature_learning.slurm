#!/bin/bash
#SBATCH --job-name=feature_learning_mw-t5-base
#SBATCH --output=feature_learning_mw-t5-base.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=shard:8
#SBATCH --mem-per-cpu=128G
#SBATCH --time=00:30:00

# testing for metaworld
python lal/feature_learning/learn_features.py --initial-loss-check \
    --data-dir=/scr/zyang966/language-preference-learning/data/metaworld --batch-size=512 \
    --use-bert-encoder  --exp-name=testing-mw-t5-base --lang-model=t5-base --traj-reg-coeff 1e-1 \
    --traj-encoder="mlp" --epochs=1 --add-norm-loss --env=metaworld --lr=1e-3 --finetune-lr=1e-3

# python lal/feature_learning/learn_features.py --initial-loss-check \
#     --data-dir=/scr/zyang966/language-preference-learning/data/metaworld --batch-size=128 \
#     --use-bert-encoder  --exp-name=testing-mw-t5-small-both-norm-loss-no-img --lang-model=t5-small --traj-reg-coeff 8e-3 \
#     --seed=6942 --traj-encoder="mlp" --epochs=1 --add-norm-loss --env=metaworld --lr=9e-4 --finetune-lr=2e-3

# --lr=5e-4 --finetune-lr=2e-3
# testing-mw...mlp uses traj-ref-coeff 1e-2, no img-obs, lr=1e-3, epochs=5, batch_size=1024
# the testing-mw... without "better accuracy" are both mlps with a traj-reg-coeff of 1e-1, no use-img-obs

# this is the best one for robosuite
# python lal/feature_learning/learn_features.py --initial-loss-check \
#     --data-dir=/scr/zyang966/language-preference-learning/data/robosuite_img_obs_res_224_more --batch-size=1024 \
#     --use-bert-encoder  --exp-name=testing-rs-t5-base-both-norm-loss --lang-model=t5-base --traj-reg-coeff 1e-1 \
#     --seed=161 --traj-encoder="mlp" --epochs=5 --add-norm-loss

# python lal/feature_learning/learn_features.py --initial-loss-check \
#     --data-dir=/scr/zyang966/language-preference-learning/data/robosuite_img_obs_res_224_more --batch-size=1024 \
#     --use-bert-encoder  --exp-name=testing-rs-t5-small --lang-model=t5-small --traj-reg-coeff 1e-2 \
#     --seed=161 --traj-encoder="mlp" --epochs=10

# python lal/feature_learning/learn_features.py --initial-loss-check \
    # --data-dir=/scr/zyang966/language-preference-learning/data/robosuite_img_obs_res_224_more --batch-size=1024 \
    # --use-bert-encoder  --exp-name=testing-rs-t5-base --lang-model=t5-base --traj-reg-coeff 1e-2 \
    # --seed=161 --traj-encoder="mlp" --epochs=10

# python lal/feature_learning/learn_features.py --initial-loss-check \
#     --data-dir=/scr/zyang966/language-preference-learning/data/robosuite_img_obs_res_224_more --batch-size=1024 \
#     --use-bert-encoder  --exp-name=testing-rs-t5-base --lang-model=t5-base --traj-reg-coeff 1e-2 \
#     --seed=161 --traj-encoder="cnn" --use-img-obs