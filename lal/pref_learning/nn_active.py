'''
This file is for active language preference learning in the Robosuite simulation environment
'''

import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
num_cpus = 4
os.environ["OMP_NUM_THREADS"] = f"{num_cpus}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_cpus}"
os.environ["MKL_NUM_THREADS"] = f"{num_cpus}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_cpus}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{num_cpus}"
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
import time
import psutil
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy

from lal.active_learning.next_traj import next_traj_method
from lal.model.encoder import NLTrajAutoencoder
from lal.pref_learning.pref_dataset import LangPrefDataset, CompPrefDataset, EvalDataset
from lal.pref_learning.utils import rs_feature_aspects, mw_feature_aspects
from lal.feature_learning.utils import LANG_MODEL_NAME, LANG_OUTPUT_DIM, AverageMeter
from lal.model_analysis.utils import get_traj_lang_embeds
from lal.model_analysis.improve_trajectory import (
    initialize_reward,
    get_feature_value,
    get_lang_feedback,
)

from data.utils import gt_reward, speed, height, distance_to_cube, distance_to_bottle
from data.utils import RS_STATE_OBS_DIM, RS_ACTION_DIM, RS_PROPRIO_STATE_DIM, RS_OBJECT_STATE_DIM
from data.utils import WidowX_STATE_OBS_DIM, WidowX_ACTION_DIM, WidowX_PROPRIO_STATE_DIM, WidowX_OBJECT_STATE_DIM
from data.utils import MW_STATE_OBS_DIM, MW_ACTION_DIM, MW_PROPRIO_STATE_DIM, MW_OBJECT_STATE_DIM

def log_cpu_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"CPU usage: {psutil.cpu_percent()}%")
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB")

# learned and true reward func (linear for now)
def init_weights_with_norm_one(m):
    if isinstance(m, nn.Linear):  # Check if the module is a linear layer
        weight_shape = m.weight.size()
        # Initialize weights with a standard method
        weights = torch.normal(mean=0, std=0.001, size=weight_shape)
        # Normalize weights to have a norm of 1
        # weights /= weights.norm(2)  # Adjust this if you need a different norm
        m.weight.data = weights
        # You can also initialize biases here if needed
        if m.bias is not None:
            m.bias.data.fill_(0)

class RewardFunc(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RewardFunc, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.apply(init_weights_with_norm_one)
        self.dropout = nn.Dropout(p=0.0)

    def forward(self, x):
        return self.dropout(self.linear(x))

def get_lang_feedback_aspect(curr_feature, reward, optimal_traj_feature, noisy=False, temperature=1.0):
    """
    Get language feedback based on feature value and true reward function

    Args:
        curr_feature: the feature value of the current trajectory
        reward: the true reward function
        optimal_traj_feature: the feature value of the optimal trajectory
        noisy: whether to add noise to the feedback
        temperature: the temperature for the softmax

    Returns:
        feature_idx: the index of the feature to give feedback on
        pos: whether the feedback is positive or negative
    """
    # potential_pos = torch.tensor(reward * (optimal_traj_feature - curr_feature.numpy()))
    # potential_neg = torch.tensor(reward * (curr_feature.numpy() - optimal_traj_feature))
    # potential = torch.cat([potential_pos, potential_neg], dim=1)

    potential = torch.tensor(reward * (optimal_traj_feature - curr_feature.numpy()))
    potential = potential / temperature
    probs = torch.softmax(potential, dim=1)
    if noisy:
        # sample a language comparison with the probabilities
        # feature_idx = torch.multinomial(probs, 1).item()
        feature_idx = np.where(np.random.multinomial(1, probs.numpy()[0]) == 1)[0][0]
    else:
        feature_idx = torch.argmax(probs).item()
    if optimal_traj_feature[feature_idx] - curr_feature[0, feature_idx] > 0:
        pos = True
    else:
        pos = False

    return feature_idx, pos

def load_data(args, split='train', DEBUG=False):
    # Load the test trajectories and language comparisons
    trajs = np.load(f"{args.data_dir}/{split}/trajs.npy")
    nlcomps = json.load(open(f"{args.data_dir}/{split}/unique_nlcomps.json", "rb"))

    nlcomp_embeds = None
    traj_img_obs = None
    actions = None

    if not args.use_bert_encoder:
        nlcomp_embeds = np.load(f"{args.data_dir}/{split}/unique_nlcomps_{args.lang_model}.npy")

    if args.use_img_obs:
        traj_img_obs = np.load(f"{args.data_dir}/{split}/traj_img_obs.npy")
        actions = np.load(f"{args.data_dir}/{split}/actions.npy")
        trajs = trajs[:, ::10, :]
        traj_img_obs = traj_img_obs[:, ::10, :]
        actions = actions[:, ::10, :]

    if DEBUG:
        print("len of trajs: " + str(len(trajs)))

    # artificially increasing dataset from 134 trajs to 968 trajs
    if args.dupe_traj >= 0 and split == "train":
        dupe_traj = args.dupe_traj
        traj_134 = trajs[dupe_traj, :, :]
        copies_traj = np.repeat(traj_134[np.newaxis, :, :], 834, axis=0)
        trajs = np.concatenate((trajs, copies_traj), axis=0)
        traj_img_134 = traj_img_obs[dupe_traj, :, :]
        copies_traj_img = np.repeat(traj_img_134[np.newaxis, :, :], 834, axis=0)
        traj_img_obs = np.concatenate((traj_img_obs, copies_traj_img), axis=0)
        actions_134 = actions[dupe_traj, :, :]
        copies_actions = np.repeat(actions_134[np.newaxis, :, :], 834, axis=0)
        actions = np.concatenate((actions, copies_actions), axis=0)

    # need to run categorize.py first to get these files
    greater_nlcomps = json.load(open(f"{args.data_dir}/train/greater_nlcomps.json", "rb"))
    less_nlcomps = json.load(open(f"{args.data_dir}/train/less_nlcomps.json", "rb"))
    classified_nlcomps = json.load(open(f"{args.data_dir}/train/classified_nlcomps.json", "rb"))
    if DEBUG:
        print("greater nlcomps size: " + str(len(greater_nlcomps)))
        print("less nlcomps size: " + str(len(less_nlcomps)))

    data = {
        "trajs": trajs,
        "nlcomps": nlcomps,
        "nlcomp_embeds": nlcomp_embeds,
        "greater_nlcomps": greater_nlcomps,
        "less_nlcomps": less_nlcomps,
        "classified_nlcomps": classified_nlcomps,
        'traj_img_obs': traj_img_obs,
        'actions': actions
    }

    return data

def get_optimal_traj(learned_reward, traj_embeds, traj_true_rewards):
    # Find the optimal trajectory with learned reward
    with torch.no_grad():
        learned_rewards = torch.tensor([learned_reward(torch.from_numpy(traj_embed)) for traj_embed in traj_embeds])
    optimal_learned_reward = traj_true_rewards[torch.argmax(learned_rewards)]
    optimal_true_reward = traj_true_rewards[torch.argmax(torch.tensor(traj_true_rewards))]
    print("chosen optimal traj idx: ", torch.argmax(learned_rewards))
    return optimal_learned_reward, optimal_true_reward

def lang_pref_learning(
    args,
    test_dataloader,
    feature_values,
    nlcomps,
    greater_nlcomps,
    less_nlcomps,
    classified_nlcomps,
    learned_reward,
    true_reward,
    traj_embeds,
    lang_embeds,
    test_traj_embeds,
    test_traj_true_rewards,
    optimal_traj_feature,
    feature_aspects,
    device,
    optimizer,
):
    # Transform numpy array to torch tensor (improve this with a function)
    traj_embeds = torch.from_numpy(traj_embeds)

    eval_cross_entropies = []
    learned_reward_norms = []
    optimal_learned_rewards, optimal_true_rewards = [], []
    all_lang_feedback = []
    all_other_language_feedback_feats = []
    logsigmoid = nn.LogSigmoid()
    init_ce = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds, device=device)
    print("Initial Cross Entropy:", init_ce)
    eval_cross_entropies.append(init_ce)

    optimal_learned_reward, optimal_true_reward = get_optimal_traj(
        learned_reward, test_traj_embeds, test_traj_true_rewards
    )

    print(f"Initial rewards: {optimal_learned_reward}, {optimal_true_reward}")
    optimal_learned_rewards.append(optimal_learned_reward)
    optimal_true_rewards.append(optimal_true_reward)

    # call next_traj_method fxn to get the iterator
    init_reward = learned_reward.linear.weight[0].detach().clone()
    init_reward /= torch.norm(init_reward)
    iterator = next_traj_method(traj_embeds, lang_embeds, init_reward, args.active, args.reward, args.lang, args.seed)

    # for it, train_lang_data in enumerate(train_lang_dataloader):
    epsilon = 1e-10
    i = 0
    score = np.inf
    # while score >= epsilon:
    # laplace_mh_reward0_traj = [25, 21, 9, 47, 110, 65, 121, 109, 24, 133, 38, 96, 117, 103, 94, 23, 78, 84, 62, 92, 127, 42, 67, 33, 79, 83, 128, 29, 14, 99, 77, 106, 102, 116, 118, 120, 113, 90, 61, 112, 72, 6, 28, 12, 115, 45, 81, 132, 76, 17]
    # laplace_mh_reward1_traj = [25, 57, 114, 34, 50, 121, 12, 55, 113, 68, 27, 28, 85, 24, 4, 132, 112, 51, 95, 77, 42, 127, 104, 123, 67, 102, 76, 29, 44, 107, 84, 14, 62, 89, 75, 87, 60, 78, 131, 116, 79, 2, 36, 9, 30, 92, 13, 8, 82, 125]
    # laplace_mh_reward2_traj = [25, 57, 114, 34, 50, 121, 35, 93, 99, 84, 75, 82, 97, 118, 109, 65, 27, 95, 6, 127, 86, 96, 28, 133, 24, 77, 132, 92, 12, 9, 101, 78, 44, 98, 56, 130, 23, 89, 4, 94, 110, 45, 21, 69, 49, 108, 13, 64, 47, 17]
    traj_reward = 0
    num_queries = 50
    while i < num_queries:
        print(f"_____________")
        print(f"Iteration: {i}")
        # sample w and l from the iterator
        # get the best query idx and the score using info gain from the iterator
        
        start_time = time.time()
        _, idx, score = iterator.next() # this takes up a lot of cpu percentage
        # score = 100 # sanity check
        # idx = laplace_mh_reward2_traj[i]
        print(f"Iterating took: {time.time() - start_time} seconds")
        # score = 100 # sanity check
        # idx = i # sanity check
        print(f"Score: {score}, Idx: {idx}")
        i += 1

        # reward weight sampling + info gain
        curr_traj_embed = traj_embeds[idx]
        curr_feature_value = torch.tensor(feature_values[idx]).unsqueeze(0).to(torch.float64)

        curr_traj_reward = torch.sum(torch.from_numpy(true_reward) * curr_feature_value.numpy())
        traj_reward += curr_traj_reward
        print(f"Curr traj reward: {curr_traj_reward.item()}")

        # check if we must still use info gain or break
        if score > epsilon:
            # Use true reward func to get language feedback (select from set)
            # First find the feature aspect to give feedback on and positive / negative
            feature_aspect_idx, pos = get_lang_feedback_aspect(
                curr_feature_value,
                true_reward,
                optimal_traj_feature,
                args.use_softmax,
                temperature=1.0,
            )
            if pos: # this simulates a user's positive feedback
                nlcomp = np.random.choice(greater_nlcomps[feature_aspects[feature_aspect_idx]])
            else: # negative feedback
                nlcomp = np.random.choice(less_nlcomps[feature_aspects[feature_aspect_idx]])

            feedback_embedding = torch.from_numpy(lang_embeds[nlcomps.index(nlcomp)]).view(1, -1)
            iterator.feed(curr_traj_embed, feedback_embedding, pos)

            all_lang_feedback.append(nlcomp)
            # Get the feature of the language comparison
            nlcomp_features = torch.concat(
                [torch.from_numpy(lang_embeds[nlcomps.index(nlcomp)]).view(1, -1) for nlcomp in all_lang_feedback],
                dim=0,
            )

            # evaluation results
            if args.use_other_feedback:
                other_nlcomps = []
                for j in range(len(feature_aspects)):
                    if j != feature_aspect_idx:
                        other_nlcomps.extend(classified_nlcomps[feature_aspects[j]])

                sampled_nlcomps = np.random.choice(other_nlcomps, args.num_other_feedback, replace=False)
                all_other_language_feedback_feats.append([lang_embeds[nlcomps.index(nlcomp)] for nlcomp in sampled_nlcomps])

                # Get the feature of the other language comparisons
                all_other_language_feedback_feats_np = np.array(all_other_language_feedback_feats)
                other_nlcomp_features = torch.concat(
                    [torch.from_numpy(feature).unsqueeze(0) for feature in all_other_language_feedback_feats_np],
                    dim=0,
                )

            learned_reward.train()
            # print(torch.norm(nlcomp_features, dim=-1))
            for k in range(args.num_iterations):
                # Compute dot product of lang(traj_opt - traj_cur)
                # Minimize the negative dot product (loss)!
                loss = -logsigmoid(learned_reward(nlcomp_features)).mean()
                # Compute preference loss of selected language feedback over other language feedback
                # now nlcomp_features is of shape N x feature_dim, need to change it to N x n x feature_dim
                # where n is the number of other language feedback
                if args.use_other_feedback:
                    nlcomp_features_expand = nlcomp_features.unsqueeze(1).expand(-1, other_nlcomp_features.shape[1], -1)
                    # Compute the cosine similarity between the language feedback and the other language feedback
                    cos_sim = F.cosine_similarity(nlcomp_features_expand, other_nlcomp_features, dim=2)
                    if args.use_constant_temp:
                        loss_lang_pref = -logsigmoid(
                            (learned_reward(nlcomp_features_expand) - learned_reward(other_nlcomp_features))
                            / args.lang_temp
                        )
                        if args.adaptive_weights:
                            weights = (1 - cos_sim) / 2
                            weights = weights / torch.sum(weights, dim=1, keepdim=True)
                            weights = weights.unsqueeze(2)
                            loss_lang_pref = torch.sum(weights * loss_lang_pref, dim=1).mean()
                        else:
                            loss_lang_pref = loss_lang_pref.mean()
                    else:
                        # Transform cosine similarity to temperature
                        # temp_cos_sim = (1 - cos_sim) / 2
                        temp_cos_sim = 1 / (1 + torch.exp(-5 * cos_sim))
                        temp_cos_sim = temp_cos_sim.unsqueeze(2)
                        # Compute the preference loss
                        loss_lang_pref = -logsigmoid(
                            (learned_reward(nlcomp_features_expand) - learned_reward(other_nlcomp_features)) / temp_cos_sim
                        ).mean()

                    loss += args.lang_loss_coeff * loss_lang_pref
                # print(loss.item())
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   

            print(f"Loss: {loss.item():.4f}, Norm of learned reward: {torch.norm(learned_reward.linear.weight):.4f}")
            learned_reward_norms.append(torch.norm(learned_reward.linear.weight).item())
            
            cross_entropy = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds, device=device)
            eval_cross_entropies.append(cross_entropy)
            optimal_learned_reward, optimal_true_reward = get_optimal_traj(
                learned_reward, test_traj_embeds, test_traj_true_rewards
            )
            print(f"Reward {i}: {optimal_learned_reward}, {optimal_true_reward}")
            print(f"Cross Entropy {i}: {cross_entropy}")
            optimal_learned_rewards.append(optimal_learned_reward)
            optimal_true_rewards.append(optimal_true_reward)

    print(f"Average Traj Reward: {traj_reward / num_queries}")
    return_dict = {
        "cross_entropy": eval_cross_entropies,
        "learned_reward_norms": learned_reward_norms,
        "optimal_learned_rewards": optimal_learned_rewards,
        "optimal_true_rewards": optimal_true_rewards,
    }
    return return_dict

def evaluate(test_dataloader, true_traj_rewards, learned_reward, traj_embeds, test=False, device="cuda"):
    """
    Evaluate the cross-entropy between the learned and true distributions

    Input:
        - test_dataloader: DataLoader for the test data
        - true_traj_rewards: true rewards of the test trajectories
        - learned_reward: the learned reward function
        - traj_embeds: the embeddings of the test trajectories
        - feature_scale_factor: the scale factor for the learned reward
        - test: whether to use the true reward for evaluation

    Output:
        - total_cross_entropy: the average cross-entropy between the learned and true distributions
    """
    total_cross_entropy = AverageMeter("cross-entropy")
    bce_loss = nn.BCELoss()
    learned_reward.eval()
    for i, test_data in enumerate(test_dataloader):
        traj_a, traj_b, idx_a, idx_b = test_data

        # get the embeddings of the two trajectories
        traj_a_embed = traj_embeds[idx_a]
        traj_b_embed = traj_embeds[idx_b]

        # get bernoulli distributions for the two trajectories
        true_rewards = torch.tensor([true_traj_rewards[idx_a], true_traj_rewards[idx_b]])
        # make true probs with softmax
        true_probs = torch.softmax(true_rewards, dim=0).float()

        if test:
            learned_probs = true_probs
        else:
            traj_a_embed = torch.tensor(traj_a_embed)
            traj_b_embed = torch.tensor(traj_b_embed)
            learned_rewards = torch.tensor([learned_reward(traj_a_embed), learned_reward(traj_b_embed)])
            learned_probs = torch.softmax(learned_rewards, dim=0)

        # calculate cross-entropy between learned and true distributions
        cross_entropy = bce_loss(learned_probs, true_probs)
        total_cross_entropy.update(cross_entropy, 1)
    return total_cross_entropy.avg

def save_results(args, results, test_ce=None, postfix="noisy"):
    save_dir = f"{args.true_reward_dir}/pref_learning"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    eval_cross_entropies = results["cross_entropy"]
    learned_reward_norms = results["learned_reward_norms"]
    optimal_learned_rewards = results["optimal_learned_rewards"]
    optimal_true_rewards = results["optimal_true_rewards"]

    result_dict = {
        "eval_cross_entropies": eval_cross_entropies,
        "learned_reward_norms": learned_reward_norms,
        "optimal_learned_rewards": optimal_learned_rewards,
        "optimal_true_rewards": optimal_true_rewards,
        "test_ce": test_ce,
    }

    np.savez(f"{save_dir}/{postfix}.npz", **result_dict)

def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load the weight of true reward
    true_reward = np.load(f"{args.true_reward_dir}/true_rewards.npy")

    # Load train data
    train_lang_data_dict = load_data(args, split='train')
    train_trajs = train_lang_data_dict["trajs"]
    train_img_obs, train_actions = train_lang_data_dict["traj_img_obs"], train_lang_data_dict["actions"]
    train_nlcomps, train_nlcomps_embed = (
        train_lang_data_dict["nlcomps"],
        train_lang_data_dict["nlcomp_embeds"],
    )
    train_greater_nlcomps, train_less_nlcomps = (
        train_lang_data_dict["greater_nlcomps"],
        train_lang_data_dict["less_nlcomps"],
    )
    train_classified_nlcomps = train_lang_data_dict["classified_nlcomps"]

    # Load test data
    test_data_dict = load_data(args, split='test')
    test_trajs = test_data_dict["trajs"]
    test_img_obs, test_actions = test_data_dict["traj_img_obs"], test_data_dict["actions"]
    test_nlcomps, test_nlcomps_embed = (
        test_data_dict["nlcomps"],
        test_data_dict["nlcomp_embeds"],
    )

    if args.env == "robosuite":
        train_feature_values = np.array([get_feature_value(traj) for traj in train_trajs])
        test_feature_values = np.array([get_feature_value(traj) for traj in test_trajs])
    elif args.env == "metaworld":
        train_feature_values = np.load(f"{args.data_dir}/train/feature_vals.npy")
        test_feature_values = np.load(f"{args.data_dir}/test/feature_vals.npy")
        train_feature_values = np.mean(train_feature_values, axis=-1)
        test_feature_values = np.mean(test_feature_values, axis=-1)
        train_feature_values = train_feature_values[:, :3]
        test_feature_values = test_feature_values[:, :3]

    all_features = np.concatenate([train_feature_values, test_feature_values], axis=0)
    feature_value_means = np.mean(all_features, axis=0)
    feature_value_stds = np.std(all_features, axis=0)

    # Normalize the feature values
    train_feature_values = (train_feature_values - feature_value_means) / feature_value_stds # around 2.04
    test_feature_values = (test_feature_values - feature_value_means) / feature_value_stds # around 2.13

    train_traj_true_rewards = np.dot(train_feature_values, true_reward)
    test_traj_true_rewards = np.dot(test_feature_values, true_reward)

    # remove trajs that have too high of a feature
    if args.dupe_traj == -1:
        train_idxs = (np.linalg.norm(train_feature_values, axis=1) < args.max_feature_norm) & (np.linalg.norm(train_feature_values, axis=1) > args.min_feature_norm)
        train_trajs = train_trajs[train_idxs]
        train_feature_values = train_feature_values[train_idxs]
        train_img_obs = train_img_obs[train_idxs]
        train_actions = train_actions[train_idxs]

        test_idxs = (np.linalg.norm(test_feature_values, axis=1) < args.max_feature_norm) & (np.linalg.norm(test_feature_values, axis=1) > args.min_feature_norm)
        test_trajs = test_trajs[test_idxs]
        test_feature_values = test_feature_values[test_idxs]
        test_img_obs = test_img_obs[test_idxs]
        test_actions = test_actions[test_idxs]

        print(train_idxs.sum(), test_idxs.sum())

    # Initialize the dataset and dataloader
    train_lang_dataset = LangPrefDataset(train_trajs, train_feature_values)
    train_lang_data = DataLoader(train_lang_dataset, batch_size=1, shuffle=True)
    test_dataset = EvalDataset(test_trajs)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Current learned language encoder
    # Load the model
    if args.use_bert_encoder:
        if 't5' in args.lang_model:
            lang_encoder = T5EncoderModel.from_pretrained(args.lang_model)
        else:
            lang_encoder = AutoModel.from_pretrained(LANG_MODEL_NAME[args.lang_model])

        tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME[args.lang_model])
        feature_dim = LANG_OUTPUT_DIM[args.lang_model]
    else:
        lang_encoder = None
        tokenizer = None
        feature_dim = 128

    if args.env == "robosuite":
        STATE_OBS_DIM = RS_STATE_OBS_DIM
        ACTION_DIM = RS_ACTION_DIM
        PROPRIO_STATE_DIM = RS_PROPRIO_STATE_DIM
        OBJECT_STATE_DIM = RS_OBJECT_STATE_DIM
    elif args.env == "widowx":
        STATE_OBS_DIM = WidowX_STATE_OBS_DIM
        ACTION_DIM = WidowX_ACTION_DIM
        PROPRIO_STATE_DIM = WidowX_PROPRIO_STATE_DIM
        OBJECT_STATE_DIM = WidowX_OBJECT_STATE_DIM
    elif args.env == "metaworld":
        STATE_OBS_DIM = MW_STATE_OBS_DIM
        ACTION_DIM = MW_ACTION_DIM
        PROPRIO_STATE_DIM = MW_PROPRIO_STATE_DIM
        OBJECT_STATE_DIM = MW_OBJECT_STATE_DIM
    else:
        raise ValueError("Invalid environment")

    model = NLTrajAutoencoder(
        STATE_OBS_DIM, ACTION_DIM, PROPRIO_STATE_DIM, OBJECT_STATE_DIM,
        encoder_hidden_dim=args.encoder_hidden_dim,
        feature_dim=feature_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        lang_encoder=lang_encoder,
        preprocessed_nlcomps=args.preprocessed_nlcomps,
        lang_embed_dim=LANG_OUTPUT_DIM[args.lang_model],
        use_bert_encoder=args.use_bert_encoder,
        traj_encoder=args.traj_encoder,
    )

    # Compatibility with old models
    state_dict = torch.load(os.path.join(args.model_dir, "best_model_state_dict.pth"))
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model.to(device) # only 135 MB

    train_traj_embeds, train_lang_embeds = get_traj_lang_embeds(
        train_trajs,
        train_nlcomps,
        model,
        device,
        args.use_bert_encoder,
        tokenizer,
        nlcomps_bert_embeds=train_nlcomps_embed,
        use_img_obs=args.use_img_obs,
        img_obs=train_img_obs,
        actions=train_actions,
        traj_encoder_type=args.traj_encoder,
    )

    test_traj_embeds, test_lang_embeds = get_traj_lang_embeds(
        test_trajs,
        test_nlcomps,
        model,
        device,
        args.use_bert_encoder,
        tokenizer,
        nlcomps_bert_embeds=test_nlcomps_embed,
        use_img_obs=args.use_img_obs,
        img_obs=test_img_obs,
        actions=test_actions,
        traj_encoder_type=args.traj_encoder,
    )

    #     # Save the embeddings
    # np.save(f"{args.data_dir}/train/traj_embeds.npy", train_traj_embeds)
    # np.save(f"{args.data_dir}/train/lang_embeds.npy", train_lang_embeds)
    # np.save(f"{args.data_dir}/test/traj_embeds.npy", test_traj_embeds)
    # np.save(f"{args.data_dir}/test/lang_embeds.npy", test_lang_embeds)
    # exit()
    
    # train_traj_embeds = np.load(f"{args.data_dir}/train/traj_embeds.npy")
    # train_traj_embeds /= np.mean(np.linalg.norm(train_traj_embeds, axis=1)) # mean
    # train_lang_embeds = np.load(f"{args.data_dir}/train/lang_embeds.npy")
    # test_traj_embeds = np.load(f"{args.data_dir}/test/traj_embeds.npy")
    # test_traj_embeds /= (np.mean(np.linalg.norm(test_traj_embeds, axis=1))) # mean
    # test_lang_embeds = np.load(f"{args.data_dir}/test/lang_embeds.npy")
    # train_traj_embeds /= 3
    # test_traj_embeds /= 3 # idk if we also need to make this thing also mean norm 1

    print("Mean Norm of Traj Embeds:", np.linalg.norm(train_traj_embeds, axis=1).mean())
    print("Mean Std of Traj Embeds:", np.std(train_traj_embeds))
    # print("Mean Norm of Lang Embeds:", np.linalg.norm(train_lang_embeds, axis=1).mean())
    # print("Mean Norm of Trajs:", np.linalg.norm(np.linalg.norm(train_trajs, axis=1), axis=1).mean())
    print("Mean Norm of Test Traj Embeds:", np.linalg.norm(test_traj_embeds, axis=1).mean())
    print("Mean Std of Test Traj Embeds:", np.std(test_traj_embeds))
    print("Norm of feature values:", np.linalg.norm(train_feature_values, axis=1).mean())
    print("Mean Std of Feature Values:", np.std(train_feature_values))
    print("Norm of test feature values:", np.linalg.norm(test_feature_values, axis=1).mean())
    print("Mean Std of Test Feature Values:", np.std(test_feature_values))

    kde_train = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(train_feature_values)
    kde_test = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(test_feature_values)
    log_density_train = kde_train.score_samples(test_feature_values)
    log_density_test = kde_test.score_samples(test_feature_values)
    p_train = np.exp(log_density_train)
    p_test = np.exp(log_density_test)
    kl_divergence = entropy(p_train, p_test)
    print("KLDiv: ", kl_divergence)
    # exit()

    # Random init learned reward
    learned_reward = RewardFunc(feature_dim, 1)
    optimizer = torch.optim.SGD(learned_reward.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    test_ce = evaluate(test_data, test_traj_true_rewards, learned_reward, test_traj_embeds, test=True, device=device)
    print(f"GT Cross Entropy: {test_ce}")

    # Load optimal trajectory given the true reward
    if args.env == "robosuite":
        optimal_traj = np.load(f"{args.true_reward_dir}/traj.npy").reshape(500, 69)
        optimal_traj_feature = get_feature_value(optimal_traj)
        
        # optimal_traj_feature = test_feature_values[np.argmax(test_traj_true_rewards)]
    elif args.env == "metaworld":
        optimal_traj = np.load(f"{args.true_reward_dir}/traj.npy").reshape(500, 46)
        optimal_traj_feature = np.load(args.true_reward_dir + "/traj_vals.npy")
        optimal_traj_feature = np.mean(optimal_traj_feature, axis=-1)
        optimal_traj_feature = optimal_traj_feature[:3]

    # Normalize the feature value
    optimal_traj_feature = (optimal_traj_feature - feature_value_means) / feature_value_stds

    if args.env == "robosuite": feature_aspects = rs_feature_aspects
    elif args.env == "metaworld": feature_aspects = mw_feature_aspects
    else: raise ValueError("Invalid environment")

    # main language learning part
    print("_____________")
    print("Noisy Pref Learning")

    noisy_results = lang_pref_learning(
        args,
        test_data,
        train_feature_values,
        train_nlcomps,
        train_greater_nlcomps,
        train_less_nlcomps,
        train_classified_nlcomps,
        learned_reward,
        true_reward,
        train_traj_embeds,
        train_lang_embeds,
        test_traj_embeds,
        test_traj_true_rewards,
        optimal_traj_feature,
        feature_aspects,
        device,
        optimizer,
    )    
        
    postfix_noisy = f"nn_active_{args.reward}_{args.lang}_lr_{args.lr}_dupe_traj_{args.dupe_traj}_num_iter_{args.num_iterations}"
    postfix_noisy += "_other_feedback_" + str(args.num_other_feedback) + "_seed_" + str(args.seed)
    postfix_noisy += f"_temp_{args.lang_temp}" + f"_lc_{args.lang_loss_coeff}" + f"_wd_{args.weight_decay}"
    save_results(args, noisy_results, test_ce=test_ce, postfix=postfix_noisy)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    # ax1.plot(noisy_results["cross_entropy"], label="Noisy")
    # ax1.plot([0, len(noisy_results["cross_entropy"])], [test_ce, test_ce], 'k--', label='Ground Truth')
    # ax1.set_xlabel("Number of Queries")
    # ax1.set_ylabel("Cross-Entropy")
    # ax1.set_title("Feedback, True Dist: Softmax")
    # ax1.legend()

    # ax2.plot(noisy_results["optimal_learned_rewards"], label="Noisy, Learned Reward")
    # ax2.plot(noisy_results["optimal_true_rewards"], label="True Reward", c="r")
    # ax2.set_xlabel("Number of Queries")
    # ax2.set_ylabel("Reward Value")
    # ax2.set_title("True Reward of Optimal Trajectory")
    # ax2.legend()

    # plt.tight_layout()
    # plt.savefig(f"{args.true_reward_dir}/pref_learning/nn_active_{args.reward}_{args.lang}_noisy_{args.seed}.png")

if __name__ == "__main__":
    torch.set_num_threads(num_cpus)
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--env", type=str, default="robosuite", help="")
    parser.add_argument("--data-dir", type=str, default="data", help="")
    parser.add_argument("--model-dir", type=str, default="models", help="")
    parser.add_argument(
        "--true-reward-dir",
        type=str,
        default="true_rewards/0",
        help="the directory of trajectories and true rewards",
    )
    parser.add_argument("--old-model", action="store_true", help="whether to use old model")
    parser.add_argument("--encoder-hidden-dim", type=int, default=128)
    parser.add_argument("--decoder-hidden-dim", type=int, default=128)
    parser.add_argument("--preprocessed-nlcomps", action="store_true", help="")
    parser.add_argument(
        "--lang-model", type=str, default="t5-small", 
        choices=["bert-base", "bert-mini", "bert-tiny", "t5-small", "t5-base"],
        help="which language model to use"
    )
    parser.add_argument(
        "--use-bert-encoder",
        action="store_true",
        help="whether to use BERT in the language encoder",
    )
    parser.add_argument("--use-img-obs", action="store_true", help="whether to use image observations")
    parser.add_argument(
        "--traj-encoder",
        default="mlp",
        choices=["mlp", "transformer", "lstm", "cnn"],
        help="which trajectory encoder to use",
    )
    parser.add_argument("--seed", type=int, default=0, help="")
    parser.add_argument(
        "--use-all-datasets",
        action="store_true",
        help="whether to use all datasets or just test set",
    )
    parser.add_argument(
        "--use-softmax",
        action="store_true",
        help="whether to use softmax or argmax for feedback",
    )
    parser.add_argument(
        "--coeff-other-feedback",
        default=1.0,
        type=float,
        help="coefficient for loss of other feedback",
    )
    parser.add_argument(
        "--lang-loss-coeff",
        default=1.0,
        type=float,
        help="coefficient for language preference loss",
    )
    parser.add_argument(
        "--adaptive-weights",
        action="store_true",
        help="whether to use adaptive weights",
    )
    parser.add_argument(
        "--method",
        default="lang",
        type=str,
        choices=["lang", "comp"],
    )
    parser.add_argument(
        "--active",
        default=4,
        type=int,
        choices=[1, 2, 3, 4],
        help="Choice for Active Learning method",
    )
    parser.add_argument(
        "--reward",
        default=1,
        type=int,
        choices=[1, 2, 3, 4],
        help="Choice for Reward sampling method",
    )
    parser.add_argument(
        "--lang",
        default=1,
        type=int,
        choices=[1, 2, 3, 4],
        help="Choice for Language sampling method",
    )
    parser.add_argument("--num-iterations", type=int, default=1, help="")
    parser.add_argument(
        "--use-other-feedback",
        action="store_true",
        help="whether to use other feedback",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="")
    parser.add_argument("--weight-decay", type=float, default=0, help="")
    parser.add_argument(
        "--num-other-feedback",
        default=1,
        type=int,
        help="number of other feedback to use",
    )
    parser.add_argument(
        "--lang-temp",
        default=1.0,
        type=float,
        help="temperature for compare with other language feedback",
    )
    parser.add_argument(
        "--use-constant-temp",
        action="store_true",
        help="whether to use constant temperature",
    )
    parser.add_argument(
        "--dupe-traj",
        default="133",
        type=int,
        help="What trajectory to duplicate to replicate a massive dataset",
    )
    parser.add_argument(
        "--max-feature-norm",
        default="2.0",
        type=float,
        help="Max norm of the trajectory feature in our dataset",
    )
    parser.add_argument(
        "--min-feature-norm",
        default="2.0",
        type=float,
        help="Min norm of the trajectory feature in our dataset",
    )
    args = parser.parse_args()
    run(args)