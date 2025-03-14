python lal/feature_learning/learn_features.py --initial-loss-check \
    --data-dir=/scr/zyang966/language-preference-learning/data/robosuite_img_obs_res_224_more --batch-size=1024 \
    --use-bert-encoder  --exp-name=testing-rs-t5-small --lang-model=t5-small --traj-reg-coeff 1e-2 \
    --seed=161 --traj-encoder="cnn" --use-img-obs --epochs=10