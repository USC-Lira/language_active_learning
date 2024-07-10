'''
This file is for active language preference learning in the Robosuite simulation environment
'''

import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
num_cpus = 16
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
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
import time

from lal.active_learning.next_traj import next_traj_method
from lal.model.encoder import NLTrajAutoencoder
from lal.pref_learning.pref_dataset import LangPrefDataset, CompPrefDataset, EvalDataset
from lal.pref_learning.utils import feature_aspects
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
        feature_idx = torch.multinomial(probs, 1).item()
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
    # learned_rewards = torch.tensor([learned_reward(torch.from_numpy(traj_embed)) for traj_embed in traj_embeds])
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
    device,
):
    # Transform numpy array to torch tensor (improve this with a function)
    traj_embeds = torch.from_numpy(traj_embeds)

    eval_cross_entropies = []
    learned_reward_norms = []
    optimal_learned_rewards, optimal_true_rewards = [], []
    init_ce = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds, device=device)
    print("Initial Cross Entropy:", init_ce)
    eval_cross_entropies.append(init_ce)

    # optimal_learned_reward, optimal_true_reward = get_optimal_traj(
    #     learned_reward, test_traj_embeds, test_traj_true_rewards
    # ) # before, check whether the learned reward fxn chose the correct traj. now, check correct + the actual reward it outputs
    learned_rewards = torch.tensor([learned_reward @ traj_embed for traj_embed in traj_embeds])
    optimal_learned_reward = learned_rewards[torch.argmax(learned_rewards)]
    true_reward_idx = torch.argmax(torch.tensor(test_traj_true_rewards))
    optimal_true_reward = test_traj_true_rewards[true_reward_idx]

    print(f"Initial rewards: {optimal_learned_reward}, {optimal_true_reward}")
    print(f"Initial reward idxs: {torch.argmax(learned_rewards)}, {true_reward_idx}")
    optimal_learned_rewards.append(optimal_learned_reward)
    optimal_true_rewards.append(optimal_true_reward)

    # call next_traj_method fxn to get the iterator
    iterator = next_traj_method(traj_embeds, lang_embeds, learned_reward, args.active, args.reward, args.lang)

    # for it, train_lang_data in enumerate(train_lang_dataloader):
    epsilon = 1e-10
    i = 0
    score = np.inf
    # import ipdb; ipdb.set_trace()
    # while score >= epsilon:
    while i < 50:
        print(f"_____________")
        print(f"Iteration: {i}")
        # sample w and l from the iterator
        # get the best query idx and the score using info gain from the iterator
        
        start_time = time.time()
        w_samples, idx, score = iterator.next() # this takes up a lot of cpu percentage
        print(f"Iterating took: {time.time() - start_time} seconds")
        print(f"Score: {score}")
        i += 1

        # reward weight sampling + info gain
        learned_reward = torch.mean(w_samples, dim=0)
        # learned_reward = w_samples[-1]
        print(f"W norm: {torch.norm(learned_reward)}")
        curr_traj_embed = traj_embeds[idx]
        curr_feature_value = feature_values[idx]

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

            # evaluation results
            learned_reward_norms.append(torch.norm(learned_reward))
            cross_entropy = evaluate(test_dataloader, test_traj_true_rewards, learned_reward, test_traj_embeds, device=device)
            eval_cross_entropies.append(cross_entropy)
            # optimal_learned_reward, optimal_true_reward = get_optimal_traj(
            #     learned_reward, test_traj_embeds, test_traj_true_rewards
            # )
            learned_rewards = torch.tensor([learned_reward @ traj_embed for traj_embed in traj_embeds])
            optimal_learned_reward = learned_rewards[torch.argmax(learned_rewards)]
            print(f"Reward {i}: {optimal_learned_reward}, {optimal_true_reward}")
            print(f"Reward {i} idxs: {torch.argmax(learned_rewards)}, {true_reward_idx}")
            optimal_learned_rewards.append(optimal_learned_reward)
            optimal_true_rewards.append(optimal_true_reward)

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
        # make true probs 0 and 1
        # true_probs = torch.tensor([torch.argmax(true_rewards) == 0, torch.argmax(true_rewards) == 1]).float()
        # make true probs with softmax
        true_probs = torch.softmax(true_rewards, dim=0).float()

        if test:
            learned_probs = true_probs

        else:
            traj_a_embed = torch.tensor(traj_a_embed)
            traj_b_embed = torch.tensor(traj_b_embed)
            # learned_rewards = torch.tensor([learned_reward(traj_a_embed), learned_reward(traj_b_embed)])
            learned_rewards = torch.tensor([learned_reward @ traj_a_embed, learned_reward @ traj_b_embed])
            learned_probs = torch.softmax(learned_rewards, dim=0)

        # calculate cross-entropy between learned and true distributions
        cross_entropy = bce_loss(learned_probs, true_probs)
        # kl_div_loss = kl_div(learned_probs.log(), true_probs)
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
    true_reward /= np.linalg.norm(true_reward)

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

    train_feature_values = np.array([get_feature_value(traj) for traj in train_trajs])

    # Load test data
    test_data_dict = load_data(args, split='test')
    test_trajs = test_data_dict["trajs"]
    test_img_obs, test_actions = test_data_dict["traj_img_obs"], test_data_dict["actions"]
    test_nlcomps, test_nlcomps_embed = (
        test_data_dict["nlcomps"],
        test_data_dict["nlcomp_embeds"],
    )

    test_feature_values = np.array([get_feature_value(traj) for traj in test_trajs])

    all_features = np.concatenate([train_feature_values, test_feature_values], axis=0)
    feature_value_means = np.mean(all_features, axis=0)
    feature_value_stds = np.std(all_features, axis=0)

    # Normalize the feature values
    train_feature_values = (train_feature_values - feature_value_means) / feature_value_stds
    test_feature_values = (test_feature_values - feature_value_means) / feature_value_stds
    # print(np.linalg.norm(test_feature_values[0]))
    # print(np.linalg.norm(true_reward))

    train_traj_true_rewards = np.dot(train_feature_values, true_reward)
    test_traj_true_rewards = np.dot(test_feature_values, true_reward)

    # Initialize the dataset and dataloader
    train_lang_dataset = LangPrefDataset(train_trajs, train_feature_values)
    train_lang_data = DataLoader(train_lang_dataset, batch_size=1, shuffle=True)
    test_dataset = EvalDataset(test_trajs)
    test_data = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Current learned language encoder
    # Load the model
    if args.use_bert_encoder:
        # if 't5' in args.lang_model:
        #     lang_encoder = T5EncoderModel.from_pretrained(args.lang_model)
        # else:
        #     lang_encoder = AutoModel.from_pretrained(LANG_MODEL_NAME[args.lang_model])

        # tokenizer = AutoTokenizer.from_pretrained(LANG_MODEL_NAME[args.lang_model])
        feature_dim = LANG_OUTPUT_DIM[args.lang_model]
    else:
        # lang_encoder = None
        # tokenizer = None
        feature_dim = 128

    # STATE_OBS_DIM = RS_STATE_OBS_DIM
    # ACTION_DIM = RS_ACTION_DIM
    # PROPRIO_STATE_DIM = RS_PROPRIO_STATE_DIM
    # OBJECT_STATE_DIM = RS_OBJECT_STATE_DIM

    # model = NLTrajAutoencoder(
    #     STATE_OBS_DIM, ACTION_DIM, PROPRIO_STATE_DIM, OBJECT_STATE_DIM,
    #     encoder_hidden_dim=args.encoder_hidden_dim,
    #     feature_dim=feature_dim,
    #     decoder_hidden_dim=args.decoder_hidden_dim,
    #     lang_encoder=lang_encoder,
    #     preprocessed_nlcomps=args.preprocessed_nlcomps,
    #     lang_embed_dim=LANG_OUTPUT_DIM[args.lang_model],
    #     use_bert_encoder=args.use_bert_encoder,
    #     traj_encoder=args.traj_encoder,
    # )

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    # model.to(device) # only 135 MB

    # # Compatibility with old models
    # state_dict = torch.load(os.path.join(args.model_dir, "best_model_state_dict.pth"), map_location=device)
    # if args.old_model:
    #     new_state_dict = {}
    #     for k, v in state_dict.items():
    #         new_k = k.replace("_hidden_layer", ".0")
    #         new_k = new_k.replace("_output_layer", ".2")
    #         new_state_dict[new_k] = v
    #     state_dict = new_state_dict

    # model.load_state_dict(state_dict)
    # model.eval()

    # train_traj_embeds, train_lang_embeds = get_traj_lang_embeds(
    #     train_trajs,
    #     train_nlcomps,
    #     model,
    #     device,
    #     args.use_bert_encoder,
    #     tokenizer,
    #     nlcomps_bert_embeds=train_nlcomps_embed,
    #     use_img_obs=args.use_img_obs,
    #     img_obs=train_img_obs,
    #     actions=train_actions
    # )
    # test_traj_embeds, test_lang_embeds = get_traj_lang_embeds(
    #     test_trajs,
    #     test_nlcomps,
    #     model,
    #     device,
    #     args.use_bert_encoder,
    #     tokenizer,
    #     nlcomps_bert_embeds=test_nlcomps_embed,
    #     use_img_obs=args.use_img_obs,
    #     img_obs=test_img_obs,
    #     actions=test_actions
    # )

    # import gc
    # del model
    # gc.collect()

    #     # Save the embeddings
    # np.save(f"{args.data_dir}/train/traj_embeds.npy", train_traj_embeds)
    # np.save(f"{args.data_dir}/train/lang_embeds.npy", train_lang_embeds)
    # np.save(f"{args.data_dir}/test/traj_embeds.npy", test_traj_embeds)
    # np.save(f"{args.data_dir}/test/lang_embeds.npy", test_lang_embeds)
    
    train_traj_embeds = np.load(f"{args.data_dir}/train/traj_embeds.npy")
    train_lang_embeds = np.load(f"{args.data_dir}/train/lang_embeds.npy")
    test_traj_embeds = np.load(f"{args.data_dir}/test/traj_embeds.npy")
    test_lang_embeds = np.load(f"{args.data_dir}/test/lang_embeds.npy")

    # Random init learned reward
    # learned_reward = RewardFunc(feature_dim, 1)
    # learned_reward_noiseless = RewardFunc(feature_dim, 1)
    # learned_reward_noiseless.linear.weight.data = learned_reward.linear.weight.data.clone()
    
    learned_reward = torch.zeros(feature_dim) # make it a vector instead of a linear layer
    learned_reward_noiseless = learned_reward.detach().clone() # make it a vector instead of a linear layer
    test_ce = evaluate(test_data, test_traj_true_rewards, learned_reward, test_traj_embeds, test=True, device=device)
    
    # Load optimal trajectory given the true reward
    optimal_traj = np.load(f"{args.true_reward_dir}/traj.npy").reshape(500, 69)
    optimal_traj_feature = get_feature_value(optimal_traj)
    # Normalize the feature value
    optimal_traj_feature = (optimal_traj_feature - feature_value_means) / feature_value_stds

    # print(train_feature_values.shape) # (135, 6)
    # print(len(train_nlcomps)) # (60)
    # print(len(train_greater_nlcomps)) # (5)
    # print(len(train_less_nlcomps)) # (5)
    # print(learned_reward.shape) # (512)
    # print(true_reward.shape) # (5)
    # print(train_traj_embeds.shape) # (134, 512)
    # print(train_lang_embeds.shape) # (60, 512)
    # print(test_traj_embeds.shape) # (34, 512)
    # print(test_traj_true_rewards.shape) # (34)
    # print(optimal_traj_feature.shape) # (5)

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
        learned_reward,
        true_reward,
        train_traj_embeds,
        train_lang_embeds,
        test_traj_embeds,
        test_traj_true_rewards,
        optimal_traj_feature,
        device,
    )    
    print("_____________")
    print("Noiseless Pref Learning")
    args.use_softmax = False
    noiseless_results = lang_pref_learning(
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
        device,
    )

    postfix_noisy = f"{args.method}_noisy_lr_{args.lr}"
    postfix_noiseless = f"{args.method}_noiseless_lr_{args.lr}"
    if args.use_other_feedback:
        postfix_noisy += "_other_feedback_" + str(args.num_other_feedback)
        postfix_noiseless += "_other_feedback_" + str(args.num_other_feedback)
        if args.use_constant_temp:
            postfix_noisy += f"_temp_{args.lang_temp}" + f"_lc_{args.lang_loss_coeff}"
            postfix_noiseless += f"_temp_{args.lang_temp}" + f"_lc_{args.lang_loss_coeff}"
        else:
            postfix_noisy += "_temp_cos" + f"_lc_{args.lang_loss_coeff}"
            postfix_noiseless += "_temp_cos" + f"_lc_{args.lang_loss_coeff}"

    else:
        postfix_noisy += "_baseline"
        postfix_noiseless += "_baseline"
    
    # Save the results in .npz files
    save_results(args, noisy_results, postfix=postfix_noisy)
    save_results(args, noiseless_results, postfix=postfix_noiseless)

    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(noisy_results["cross_entropy"], label="Noisy")
    ax1.plot(noiseless_results["cross_entropy"], label="Noiseless")
    ax1.plot([0, len(noisy_results["cross_entropy"])], [test_ce, test_ce], 'k--', label='Ground Truth')
    ax1.set_xlabel("Number of Queries")
    ax1.set_ylabel("Cross-Entropy")
    ax1.set_title("Feedback, True Dist: Softmax")
    ax1.legend()

    ax2.plot(noisy_results["optimal_learned_rewards"], label="Noisy, Learned Reward")
    ax2.plot(noiseless_results["optimal_learned_rewards"], label="Noiseless, Learned Reward")
    ax2.plot(noisy_results["optimal_true_rewards"], label="True Reward", c="r")
    ax2.set_xlabel("Number of Queries")
    ax2.set_ylabel("Reward Value")
    # ax2.set_title("True Reward of Optimal Trajectory")
    ax2.set_title("Mean Reward of Trajectories")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"{args.true_reward_dir}/pref_learning/{args.method}_{args.reward}_{args.lang}_pref.png")

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

    args = parser.parse_args()
    run(args)