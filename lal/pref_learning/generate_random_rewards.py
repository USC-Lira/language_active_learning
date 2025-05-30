"""Generate True Rewards for the Preference Learning Task with Cross-Entropy Less than 0.4"""


import numpy as np
import argparse
import torch
from torch.utils.data import DataLoader

from model_analysis.improve_trajectory import initialize_reward
from pref_learning.active_sim_pb import load_data, get_feature_value, evaluate_ce
from pref_learning.pref_dataset import EvalDataset

true_reward_dim = 5

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='/scr/eisukehirota/language-preference-learning/data/data_pref_learning', help='')
parser.add_argument('--lang-model', type=str, default='t5-small', help='which BERT model to use')
parser.add_argument('--use-bert-encoder', type=bool, default=True)
parser.add_argument('--use-img-obs', type=bool, default=True)
args = parser.parse_args()

# Load the data
train_data_dict = load_data(args)
train_trajs = train_data_dict['trajs']
train_nlcomps, train_nlcomps_embed = train_data_dict['nlcomps'], train_data_dict['nlcomp_embeds']
train_greater_nlcomps, train_less_nlcomps = train_data_dict['greater_nlcomps'], train_data_dict['less_nlcomps']
train_classified_nlcomps = train_data_dict['classified_nlcomps']
train_feature_values = np.array([get_feature_value(traj) for traj in train_trajs])

test_data_dict = load_data(args, split="test")
test_trajs = test_data_dict['trajs']
test_nlcomps, test_nlcomps_embed = test_data_dict['nlcomps'], test_data_dict['nlcomp_embeds']
test_feature_values = np.array([get_feature_value(traj) for traj in test_trajs])
test_dataset = EvalDataset(test_trajs)
test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)

all_features = np.concatenate([train_feature_values, test_feature_values], axis=0)
feature_value_means = np.mean(all_features, axis=0)
feature_value_stds = np.std(all_features, axis=0)

# Normalize the feature values
train_feature_values = (train_feature_values - feature_value_means) / feature_value_stds
test_feature_values = (test_feature_values - feature_value_means) / feature_value_stds

test_traj_embeds = np.load(f"{args.data_dir}/test/traj_embeds.npy")
test_traj_embeds /= (np.mean(np.linalg.norm(test_traj_embeds, axis=1))) # mean

true_rewards = []
for _ in range(1):
    cross_entropy = 1.0
    true_reward = None
    while cross_entropy > 0.2:
        true_reward = initialize_reward(true_reward_dim)
        print(true_reward)
        true_traj_rewards = test_feature_values @ true_reward
        test_entropy = evaluate_ce(test_data, true_traj_rewards, true_reward, test_traj_embeds, test=True)
        cross_entropy = test_entropy
        print(cross_entropy)
    true_rewards.append(true_reward)

np.save('/scr/eisukehirota/language_active_learning/lal/pref_learning/true_rewards/5/true_rewards.npy', true_rewards[0])
print(true_rewards)