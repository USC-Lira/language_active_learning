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
        weights = torch.normal(mean=0, std=0.1, size=weight_shape) # 0.1 for t5-base, 0.001 for t5-small
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

    def forward(self, x):
        return self.linear(x)

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

    # print(optimal_traj_feature.shape) # (5,)
    # print(curr_feature.shape) # (5,)
    potential = torch.tensor(reward * (optimal_traj_feature - curr_feature))
    potential = potential / temperature
    probs = torch.softmax(potential, dim=0)
    if noisy:
        # sample a language comparison with the probabilities
        # feature_idx = torch.multinomial(probs, 1).item()
        feature_idx = np.where(np.random.multinomial(1, probs.numpy()) == 1)[0][0]
    else:
        feature_idx = torch.argmax(probs).item()
    if optimal_traj_feature[feature_idx] - curr_feature[feature_idx] > 0:
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

    ###### if you'd like to add extra data ######
    # if args.env == "metaworld":
    #     if split == "test":
    #         trajs_extra = np.load(f"{args.data_dir_extra}/{split}/trajs.npy")
    #         trajs_extra = trajs_extra[:, ::10, :]
    #         # print(trajs.shape, trajs_extra.shape)
    #         trajs = np.concatenate((trajs, trajs_extra))

    #     traj_img_obs_extra = None
    #     actions_extra = None

    #     if args.use_img_obs:
    #         traj_img_obs_extra = np.load(f"{args.data_dir_extra}/{split}/traj_img_obs.npy")
    #         actions_extra = np.load(f"{args.data_dir_extra}/{split}/actions.npy")
    #         traj_img_obs_extra = traj_img_obs_extra[:, ::10, :]
    #         actions_extra = actions_extra[:, ::10, :]

    #     traj_img_obs = np.concatenate((traj_img_obs, traj_img_obs_extra))
    #     actions = np.concatenate((actions, actions_extra))
        # print(trajs.shape, traj_img_obs.shape, actions.shape)

    ###### instead, if you'd like to duplicate one of the trajectories ######
    # if args.dupe_traj >= 0 and split == "train":
    #     dupe_traj = args.dupe_traj
    #     traj_134 = trajs[dupe_traj, :, :]
    #     copies_traj = np.repeat(traj_134[np.newaxis, :, :], 834, axis=0)
    #     trajs = np.concatenate((trajs, copies_traj), axis=0)
    #     traj_img_134 = traj_img_obs[dupe_traj, :, :]
    #     copies_traj_img = np.repeat(traj_img_134[np.newaxis, :, :], 834, axis=0)
    #     traj_img_obs = np.concatenate((traj_img_obs, copies_traj_img), axis=0)
    #     actions_134 = actions[dupe_traj, :, :]
    #     copies_actions = np.repeat(actions_134[np.newaxis, :, :], 834, axis=0)
    #     actions = np.concatenate((actions, copies_actions), axis=0)

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
    learned_rewards = torch.tensor([learned_reward @ torch.from_numpy(traj_embed) for traj_embed in traj_embeds])
    optimal_learned_reward = traj_true_rewards[torch.argmax(learned_rewards)]
    optimal_true_reward = traj_true_rewards[torch.argmax(torch.tensor(traj_true_rewards))]
    return optimal_learned_reward, optimal_true_reward

def lang_pref_learning(
    args,
    test_dataloader,
    feature_values,
    nlcomps,
    greater_nlcomps,
    less_nlcomps,
    learned_reward,
    true_reward,
    traj_embeds,
    lang_embeds,
    test_traj_embeds,
    test_traj_true_rewards,
    optimal_traj_feature,
    feature_aspects,
    device,
):
    traj_embeds = torch.from_numpy(traj_embeds)

    eval_cross_entropies = []
    learned_reward_norms = []
    optimal_learned_rewards, optimal_true_rewards = [], []
    init_ce = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds, device=device)
    print("Initial Cross Entropy:", init_ce)
    learned_reward /= torch.norm(learned_reward)
    eval_cross_entropies.append(init_ce)

    optimal_learned_reward, optimal_true_reward = get_optimal_traj(
        learned_reward, test_traj_embeds, test_traj_true_rewards
    )

    optimal_learned_rewards.append(optimal_learned_reward)
    optimal_true_rewards.append(optimal_true_reward)

    # call next_traj_method fxn to get the iterator
    iterator = next_traj_method(traj_embeds, lang_embeds, learned_reward, args.active, args.reward, args.lang, args.seed)

    epsilon = 1e-10
    i = 0
    score = np.inf
    while i < 50:        
        w_samples, idx, score = iterator.next()
        i += 1

        learned_reward = torch.mean(w_samples, dim=0).to(torch.float)
        curr_traj_embed = traj_embeds[idx]
        curr_feature_value = feature_values[idx]

        # check if we must still use info gain or break
        if score > epsilon:
            # Use true reward func to get language feedback (select from set)
            # First find the feature aspect to give feedback on and positive / negative
            if args.env == "robosuite": temp = 1.0
            elif args.env == "metaworld": temp = 1.0
            feature_aspect_idx, pos = get_lang_feedback_aspect(
                curr_feature_value,
                true_reward,
                optimal_traj_feature,
                args.use_softmax,
                temperature=temp,
            )
            if pos: # this simulates a user's positive feedback
                nlcomp = np.random.choice(greater_nlcomps[feature_aspects[feature_aspect_idx]])
            else: # negative feedback
                nlcomp = np.random.choice(less_nlcomps[feature_aspects[feature_aspect_idx]])

            feedback_embedding = torch.from_numpy(lang_embeds[nlcomps.index(nlcomp)]).view(1, -1)
            iterator.feed(curr_traj_embed, feedback_embedding, pos)

            # evaluation results
            learned_reward_norms.append(torch.norm(learned_reward))
            cross_entropy = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds, device=device)
            eval_cross_entropies.append(cross_entropy)
            optimal_learned_reward, optimal_true_reward = get_optimal_traj(
                learned_reward, test_traj_embeds, test_traj_true_rewards
            )
            optimal_learned_rewards.append(optimal_learned_reward)
            optimal_true_rewards.append(optimal_true_reward)
            # if i == 10: print(cross_entropy)

    print(f"Final CE: {cross_entropy}")
    print(f"Final Reward: {optimal_learned_reward/optimal_true_reward}")
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
            # learned_rewards = true_rewards
        else:
            traj_a_embed = torch.tensor(traj_a_embed)
            traj_b_embed = torch.tensor(traj_b_embed)
            learned_rewards = torch.tensor([learned_reward.to(torch.float) @ traj_a_embed.to(torch.float), learned_reward.to(torch.float) @ traj_b_embed.to(torch.float)])
            learned_probs = torch.softmax(learned_rewards, dim=0).float()

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

        #### if you are adding more data ####
        # train_feature_values_extra = np.load(f"{args.data_dir_extra}/train/feature_vals.npy")
        # test_feature_values_extra = np.load(f"{args.data_dir_extra}/test/feature_vals.npy")
        # train_feature_values_extra = np.mean(train_feature_values_extra, axis=-1)
        # test_feature_values_extra = np.mean(test_feature_values_extra, axis=-1)
        # train_feature_values_extra = train_feature_values_extra[:, :3]
        # test_feature_values_extra = test_feature_values_extra[:, :3]
        # train_feature_values = np.concatenate((train_feature_values, train_feature_values_extra))
        # test_feature_values = np.concatenate((test_feature_values, test_feature_values_extra))

    all_features = np.concatenate([train_feature_values, test_feature_values], axis=0)
    feature_value_means = np.mean(all_features, axis=0)
    feature_value_stds = np.std(all_features, axis=0)

    # Normalize the feature values
    train_feature_values = (train_feature_values - feature_value_means) / feature_value_stds # around 2.04
    test_feature_values = (test_feature_values - feature_value_means) / feature_value_stds # around 2.13

    train_traj_true_rewards = np.dot(train_feature_values, true_reward)
    test_traj_true_rewards = np.dot(test_feature_values, true_reward)

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
    model.to(device)

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

    # skewing the dataset if applicable
    if args.dupe_traj == -1 and args.env == "metaworld":
        closest_n = 90
        farthest_n = 10
        mean_train_traj_embeds = np.mean(train_traj_embeds, axis=0)

        closest_idcs = np.argsort(np.linalg.norm(train_traj_embeds - mean_train_traj_embeds, axis=1))[:closest_n]
        farthest_idcs = np.argsort(-np.linalg.norm(train_traj_embeds - mean_train_traj_embeds, axis=1))[:farthest_n]
        train_idxs = np.concatenate((closest_idcs, farthest_idcs))
        train_trajs = train_trajs[train_idxs]
        train_feature_values = train_feature_values[train_idxs]
        train_traj_embeds = train_traj_embeds[train_idxs]
        if args.use_img_obs:
            train_img_obs = train_img_obs[train_idxs] 
            train_actions = train_actions[train_idxs]

    # Initialize the dataset and dataloader
    train_lang_dataset = LangPrefDataset(train_trajs, train_feature_values)
    train_lang_data = DataLoader(train_lang_dataset, batch_size=1, shuffle=True)
    test_dataset = EvalDataset(test_trajs)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)

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

    # Random init learned reward
    learned_reward = RewardFunc(feature_dim, 1)
    learned_reward = learned_reward.linear.weight[0].detach().clone() # make it a vector instead of a linear layer
    test_ce = evaluate(test_data, test_traj_true_rewards, learned_reward, test_traj_embeds, test=True, device=device)

    # Load optimal trajectory given the true reward
    if args.env == "robosuite":
        if int(args.true_reward_dir[-1]) <= 2:
            optimal_traj = np.load(f"{args.true_reward_dir}/traj.npy").reshape(500, 69)
            optimal_traj_feature = get_feature_value(optimal_traj)
        else:
            optimal_traj_feature = np.load(args.true_reward_dir + "/feature_vals.npy")
    elif args.env == "metaworld":
        if int(args.true_reward_dir[-1]) < 2:
            optimal_traj_feature = np.load(args.true_reward_dir + "/traj_vals.npy")
            optimal_traj_feature = np.mean(optimal_traj_feature, axis=-1)
            optimal_traj_feature = optimal_traj_feature[:3]
        else:
            optimal_traj_feature = np.load(args.true_reward_dir + "/feature_vals.npy")

    # Normalize the feature value
    optimal_traj_feature = (optimal_traj_feature - feature_value_means) / feature_value_stds

    if args.env == "robosuite": feature_aspects = rs_feature_aspects
    elif args.env == "metaworld": feature_aspects = mw_feature_aspects
    else: raise ValueError("Invalid environment")

    # main language learning part
    print("_____________")
    print("Active")
    print(f"reward={args.true_reward_dir[-1]}, seed={args.seed}")
    print(f"GT Cross Entropy: {test_ce}")
    noisy_results = lang_pref_learning(
        args,
        test_data,
        train_feature_values,
        train_nlcomps,
        train_greater_nlcomps,
        train_less_nlcomps,
        learned_reward,
        true_reward,
        train_traj_embeds,
        train_lang_embeds,
        test_traj_embeds,
        test_traj_true_rewards,
        optimal_traj_feature,
        feature_aspects,
        device,
    )    

    postfix_noisy = f"{args.active}_{args.reward}_{args.lang}_dupe_traj_{args.dupe_traj}_seed_{args.seed}"
    save_results(args, noisy_results, test_ce=test_ce, postfix=postfix_noisy)

if __name__ == "__main__":
    torch.set_num_threads(num_cpus)
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--env", type=str, default="robosuite", help="")
    parser.add_argument("--data-dir", type=str, default="data", help="")
    parser.add_argument("--data-dir-extra", type=str, default="data", help="")
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
    parser.add_argument(
        "--dupe-traj",
        default="133",
        type=int,
        help="What trajectory to duplicate to replicate a massive dataset",
    )

    args = parser.parse_args()
    run(args)