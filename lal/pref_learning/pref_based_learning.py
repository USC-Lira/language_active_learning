# preference based learning

# Importing the libraries
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy

from transformers import AutoModel, AutoTokenizer, T5EncoderModel

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

# learned and true reward func (linear for now)
def init_weights_with_norm_one(m):
    if isinstance(m, nn.Linear):  # Check if the module is a linear layer
        weight_shape = m.weight.size()
        # Initialize weights with a standard method
        weights = torch.normal(mean=0, std=0.1, size=weight_shape) # 0.1 for RS, 0.001 for MW was default
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

    ###### if you'd like to add extra data ######
    # if args.env == "metaworld":
    #     if split == "test":
    #         trajs_extra = np.load(f"{args.data_dir_extra}/{split}/trajs.npy")
    #         if args.use_img_obs:
    #             trajs_extra = trajs_extra[:, ::10, :]
    #         # print(trajs.shape, trajs_extra.shape)
    #         trajs = np.concatenate((trajs, trajs_extra))

    #     traj_img_obs_extra = None
    #     actions_extra = None

    #     if args.use_img_obs:
    #         traj_img_obs_extra = np.load(f"{args.data_dir_extra}/{split}/traj_img_obs.npy")
    #         actions_extra = np.load(f"{args.data_dir_extra}/{split}/actions.npy")
    #         traj_img_obs_extra = traj_img_obs_extra[:, ::10, :]
    #         actions_extra = actions_extra[:, ::10, :]
    #         traj_img_obs = np.concatenate((traj_img_obs, traj_img_obs_extra))
    #         actions = np.concatenate((actions, actions_extra))
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

def reconstruct_traj(traj_embeds, model, nlcomp_embeds):
    new_trajs = torch.from_numpy(traj_embeds) + nlcomp_embeds
    recon_trajs = model.traj_decoder(new_trajs.to("cuda"))
    return recon_trajs.detach().cpu().numpy()

def get_optimal_traj(learned_reward, traj_embeds, traj_true_rewards):
    # Fine the optimal trajectory with learned reward
    learned_rewards = torch.tensor([learned_reward(torch.from_numpy(traj_embed)) for traj_embed in traj_embeds])
    optimal_learned_reward = traj_true_rewards[torch.argmax(learned_rewards)]
    optimal_true_reward = traj_true_rewards[torch.argmax(torch.tensor(traj_true_rewards))]

    return optimal_learned_reward, optimal_true_reward

def comp_pref_learning(
    args,
    train_lang_dataloader,
    test_dataloader,
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
    optimizer,
    lr_scheduler=None,
    DEBUG=False,
):
    # TODO: write a new train dataloader
    # Transform numpy array to torch tensor (improve this with a function)
    traj_embeds = torch.from_numpy(traj_embeds)

    eval_cross_entropies = []
    learned_reward_norms = []
    sampled_traj_a_embeds, sampled_traj_b_embeds = [], []
    sampled_labels = []
    optimal_learned_rewards, optimal_true_rewards = [], []
    logsigmoid = nn.LogSigmoid()
    init_ce = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds)
    print("Initial Cross Entropy:", init_ce)
    eval_cross_entropies.append(init_ce)

    optimal_learned_reward, optimal_true_reward = get_optimal_traj(
        learned_reward, test_traj_embeds, test_traj_true_rewards
    )
    # print(f"Initial rewards: {optimal_learned_reward}, {optimal_true_reward}")
    optimal_learned_rewards.append(optimal_learned_reward)
    optimal_true_rewards.append(optimal_true_reward)
    CEloss = nn.CrossEntropyLoss()

    for it, train_lang_data in enumerate(train_lang_dataloader):
        if it > 49:
            break
        traj_a, traj_b, feature_a, feature_b, idx_a, idx_b = train_lang_data
        traj_a_embed, traj_b_embed = traj_embeds[idx_a], traj_embeds[idx_b]
        sampled_traj_a_embeds.append(traj_a_embed.view(1, -1))
        sampled_traj_b_embeds.append(traj_b_embed.view(1, -1))

        true_reward_a = torch.sum(torch.from_numpy(true_reward) * feature_a.numpy())
        true_reward_b = torch.sum(torch.from_numpy(true_reward) * feature_b.numpy())

        sampled_labels.append(true_reward_a < true_reward_b)

        for i in range(args.num_iterations):
            pred_rewards_a = learned_reward(torch.concat(sampled_traj_a_embeds, dim=0))
            pred_rewards_b = learned_reward(torch.concat(sampled_traj_b_embeds, dim=0))
            pred_rewards = torch.cat([pred_rewards_a, pred_rewards_b], dim=-1)

            labels = torch.tensor(sampled_labels).long()

            assert pred_rewards.shape[0] == labels.shape[0]

            loss = CEloss(pred_rewards, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        learned_reward_norms.append(torch.norm(learned_reward.linear.weight).item())
        
        # Evaluation
        cross_entropy = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds)
        eval_cross_entropies.append(cross_entropy)
        optimal_learned_reward, optimal_true_reward = get_optimal_traj(
            learned_reward, test_traj_embeds, test_traj_true_rewards
        )
        optimal_learned_rewards.append(optimal_learned_reward)
        optimal_true_rewards.append(optimal_true_reward)
        # if it == 10: print(cross_entropy)

    print(f"Final CE: {cross_entropy}")
    print(f"Final Reward: {optimal_learned_reward/optimal_true_reward}")
    return_dict = {
        "cross_entropy": eval_cross_entropies,
        "learned_reward_norms": learned_reward_norms,
        "optimal_learned_rewards": optimal_learned_rewards,
        "optimal_true_rewards": optimal_true_rewards,
    }
    return return_dict

def lang_pref_learning(
    args,
    train_lang_dataloader,
    test_dataloader,
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
    optimizer,
    feature_aspects,
    lr_scheduler=None,
    DEBUG=False,
):
    # Transform numpy array to torch tensor (improve this with a function)
    traj_embeds = torch.from_numpy(traj_embeds)

    eval_cross_entropies = []
    learned_reward_norms = []
    all_lang_feedback = []
    all_other_language_feedback_feats = []
    sampled_traj_embeds = []
    sampled_traj_true_rewards = []
    optimal_learned_rewards, optimal_true_rewards = [], []
    logsigmoid = nn.LogSigmoid()
    init_ce = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds)
    print("Initial Cross Entropy:", init_ce)
    eval_cross_entropies.append(init_ce)

    optimal_learned_reward, optimal_true_reward = get_optimal_traj(
        learned_reward, test_traj_embeds, test_traj_true_rewards
    )
    optimal_learned_rewards.append(optimal_learned_reward)
    optimal_true_rewards.append(optimal_true_reward)

    traj_reward = 0
    num_queries = 49
    for it, train_lang_data in enumerate(train_lang_dataloader):
        if it > num_queries:
            break
        curr_traj, curr_feature_value, idx = train_lang_data
        curr_traj_embed = traj_embeds[idx]
        sampled_traj_embeds.append(curr_traj_embed.view(1, -1))
        curr_traj_reward = torch.sum(torch.from_numpy(true_reward) * curr_feature_value.numpy())
        sampled_traj_true_rewards.append(curr_traj_reward.view(1, -1))

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
        if pos:
            nlcomp = np.random.choice(greater_nlcomps[feature_aspects[feature_aspect_idx]])
        else:
            nlcomp = np.random.choice(less_nlcomps[feature_aspects[feature_aspect_idx]])

        all_lang_feedback.append(nlcomp)
        # Get the feature of the language comparison
        nlcomp_features = torch.concat(
            [torch.from_numpy(lang_embeds[nlcomps.index(nlcomp)]).view(1, -1) for nlcomp in all_lang_feedback],
            dim=0,
        )

        # Randomly sample feedback for other features in the training set
        if args.use_other_feedback:
            other_nlcomps = []
            for i in range(len(feature_aspects)):
                if i != feature_aspect_idx:
                    other_nlcomps.extend(classified_nlcomps[feature_aspects[i]])

            sampled_nlcomps = np.random.choice(other_nlcomps, args.num_other_feedback, replace=False)
            all_other_language_feedback_feats.append([lang_embeds[nlcomps.index(nlcomp)] for nlcomp in sampled_nlcomps])

            # Get the feature of the other language comparisons
            all_other_language_feedback_feats_np = np.array(all_other_language_feedback_feats)
            other_nlcomp_features = torch.concat(
                [torch.from_numpy(feature).unsqueeze(0) for feature in all_other_language_feedback_feats_np],
                dim=0,
            )

        learned_reward.train()
        for i in range(args.num_iterations):
            # Compute dot product of lang(traj_opt - traj_cur)
            # Minimize the negative dot product (loss)!
            loss = -logsigmoid(learned_reward(nlcomp_features)).mean()
            # loss = 0 # for implicit ablation

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
            if lr_scheduler is not None:
                lr_scheduler.step()
                
        learned_reward_norms.append(torch.norm(learned_reward.linear.weight).item())

        # Evaluation
        cross_entropy = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds)
        eval_cross_entropies.append(cross_entropy)
        optimal_learned_reward, optimal_true_reward = get_optimal_traj(
            learned_reward, test_traj_embeds, test_traj_true_rewards
        )
        optimal_learned_rewards.append(optimal_learned_reward)
        optimal_true_rewards.append(optimal_true_reward)
        # if it == 10: print(cross_entropy)

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
    learned_reward.eval()
    for i, test_data in enumerate(test_dataloader):
        traj_a, traj_b, idx_a, idx_b = test_data

        # get bernoulli distributions for the two trajectories
        true_rewards = torch.tensor([true_traj_rewards[idx_a], true_traj_rewards[idx_b]])
        true_probs = torch.softmax(true_rewards, dim=0).float()

        if test:
            learned_probs = true_probs
            # learned_rewards = true_rewards

        else:
            # get the embeddings of the two trajectories
            traj_a_embed = traj_embeds[idx_a]
            traj_b_embed = traj_embeds[idx_b]
            
            traj_a_embed = torch.tensor(traj_a_embed)
            traj_b_embed = torch.tensor(traj_b_embed)
            learned_rewards = torch.tensor([learned_reward(traj_a_embed), learned_reward(traj_b_embed)])
            learned_probs = torch.softmax(learned_rewards, dim=0)

        # calculate cross-entropy between learned and true distributions
        cross_entropy = bce_loss(learned_probs, true_probs)
        total_cross_entropy.update(cross_entropy, 1)
    return total_cross_entropy.avg


def save_results(args, results, postfix="noisy"):
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

    # gt_reward, speed, height, distance_to_cube, distance_to_bottle # rs
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
    model.to(device)

    # Check if the embeddings are already computed
    # If not, compute the embeddings
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
    # train_lang_data = DataLoader(train_lang_dataset, batch_size=1, shuffle=False) # sanity check
    train_comp_dataset = CompPrefDataset(train_trajs, train_feature_values)
    train_comp_data = DataLoader(train_comp_dataset, batch_size=1, shuffle=True)
    test_dataset = EvalDataset(test_trajs)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Random init learned reward
    learned_reward = RewardFunc(feature_dim, 1)
    optimizer = torch.optim.SGD(learned_reward.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    test_ce = evaluate(test_data, test_traj_true_rewards, learned_reward, test_traj_embeds, test=True)

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

    if args.method == "comp":
        print("_____________")
        print("Comp Learning")
        print(f"lr={args.lr}, reward={args.true_reward_dir[-1]}, seed={args.seed}")
        print(f"GT Cross Entropy: {test_ce}")
        noisy_results = comp_pref_learning(
            args,
            train_comp_data,
            test_data,
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
            optimizer,
            feature_aspects,
        )

    elif args.method == "lang":
        print("_____________")
        print("Lang Learning")
        print(f"lr={args.lr}, reward={args.true_reward_dir[-1]}, seed={args.seed}")
        print(f"GT Cross Entropy: {test_ce}")
        noisy_results = lang_pref_learning(
            args,
            train_lang_data,
            test_data,
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
            optimizer,
            feature_aspects,
        )
    
    else:
        raise ValueError("Invalid method")
    
    postfix_noisy = f"{args.method}_noisy_lr_{args.lr}_dupe_traj_{args.dupe_traj}_num_iter_{args.num_iterations}"
    # postfix_noisy = f"{args.method}_noisy_lr_{args.lr}_exp" # ablation, exp, 0 num_feedback
    # postfix_noisy = f"{args.method}_noisy_lr_{args.lr}_imp" # ablation, imp, 20 num_feedback but no base loss
    postfix_noisy += "_other_feedback_" + str(args.num_other_feedback) + "_seed_" + str(args.seed)
    postfix_noisy += f"_temp_{args.lang_temp}" + f"_lc_{args.lang_loss_coeff}" + f"_wd_{args.weight_decay}"
    save_results(args, noisy_results, postfix=postfix_noisy)

if __name__ == "__main__":
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
    parser.add_argument("--weight-decay", type=float, default=0, help="")
    parser.add_argument("--lr", type=float, default=1e-3, help="")
    parser.add_argument("--seed", type=int, default=0, help="")
    parser.add_argument("--num-iterations", type=int, default=1, help="")
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
        "--use-other-feedback",
        action="store_true",
        help="whether to use other feedback",
    )
    parser.add_argument(
        "--num-other-feedback",
        default=1,
        type=int,
        help="number of other feedback to use",
    )
    parser.add_argument(
        "--coeff-other-feedback",
        default=1.0,
        type=float,
        help="coefficient for loss of other feedback",
    )
    parser.add_argument(
        "--use-constant-temp",
        action="store_true",
        help="whether to use constant temperature",
    )
    parser.add_argument(
        "--lang-temp",
        default=1.0,
        type=float,
        help="temperature for compare with other language feedback",
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
