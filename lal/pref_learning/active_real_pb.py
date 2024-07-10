"""
Preference learning for real robot experiments
"""

# Importing the libraries
import json
import pickle
import re
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer, T5EncoderModel
from lang_pref_learning.model.encoder import NLTrajAutoencoder
from lang_pref_learning.pref_learning.pref_dataset import LangPrefDataset, CompPrefDataset, EvalDataset
from lang_pref_learning.pref_learning.utils import feature_aspects
from lang_pref_learning.feature_learning.utils import LANG_MODEL_NAME, LANG_OUTPUT_DIM, AverageMeter
from lang_pref_learning.real_robot_exp.utils import get_traj_embeds_wx, get_lang_embed, get_traj_lang_embeds
from lang_pref_learning.real_robot_exp.improve_trajectory import get_feature_value
from lang_pref_learning.real_robot_exp.utils import replay_traj_widowx, replay_trajectory_video, remove_special_characters
from data.utils import WidowX_STATE_OBS_DIM, WidowX_ACTION_DIM, WidowX_PROPRIO_STATE_DIM, WidowX_OBJECT_STATE_DIM
optimal_candidates = [5, 8, 10, 15, 18, 20, 21, 22, 23, 25, 27, 29]

try:
    import time
    import rospy
    import pickle as pkl

    from multicam_server.topic_utils import IMTopic
    from widowx_envs.widowx_env import WidowXEnv

    env_params = {
    'camera_topics': [IMTopic('/blue/image_raw')
                      ],
    'gripper_attached': 'custom',
    'skip_move_to_neutral': True,
    'move_to_rand_start_freq': -1,
    'fix_zangle': 0.1,
    'move_duration': 0.2,
    'adaptive_wait': True,
    'action_clipping': None
}
    
except ImportError as e:
    print(e)

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

def load_data(args, split='train', DEBUG=False):
    # Load the test trajectories and language comparisons
    trajs = np.load(f"{args.data_dir}/{split}/trajs.npy")
    nlcomps = json.load(open(f"{args.data_dir}/{split}/unique_nlcomps.json", "rb"))

    nlcomp_embeds = None
    traj_img_obs = None

    if not args.use_bert_encoder:
        nlcomp_embeds = np.load(f"{args.data_dir}/{split}/unique_nlcomps_{args.lang_model}.npy")

    if args.use_img_obs:
        traj_img_obs = np.load(f"{args.data_dir}/{split}/traj_img_obs.npy")
        trajs = trajs[:, ::10, :]
        traj_img_obs = traj_img_obs[:, ::10, :]

    if DEBUG:
        print("len of trajs: " + str(len(trajs)))

    data = {
        "trajs": trajs,
        "nlcomps": nlcomps,
        "nlcomp_embeds": nlcomp_embeds,
        # "greater_nlcomps": greater_nlcomps,
        # "less_nlcomps": less_nlcomps,
        # "classified_nlcomps": classified_nlcomps,
        'traj_img_obs': traj_img_obs,
        # 'actions': actions
    }
    return data

def get_optimal_traj(learned_reward, traj_embeds):
    # Fine the optimal trajectory with learned reward
    learned_rewards = learned_reward(traj_embeds)
    optim_traj_idx = torch.argmax(learned_rewards)

    return optim_traj_idx

def lang_pref_learning(
    args,
    # dataloader,
    trajs,
    feature_values,
    model,
    nlcomps, # None
    greater_nlcomps, # None
    less_nlcomps, # None
    classified_nlcomps, # None
    learned_reward,
    traj_embeds,
    traj_img_obs,
    lang_embeds,
    optimizer,
    device,
    tokenizer,
    lang_encoder,
    lr_scheduler=None,
    DEBUG=False,
):
    # Transform numpy array to torch tensor (improve this with a function)
    traj_embeds = torch.from_numpy(traj_embeds)

    all_lang_feedback = []
    all_other_language_feedback_feats = []
    all_lang_embeds = []
    optim_traj_scores = []
    logsigmoid = nn.LogSigmoid()

    optim_candidates_embeds = traj_embeds[optimal_candidates]
    optim_candidates_images = [traj_img_obs[i] for i in optimal_candidates]

    # if args.active==1: then do info gain on both language and reward
    # if args.active==2: then do info gain on reward
    # if args.active==3: then do info gain on reward
    # if args.active==4: then do enumerate

    for it, train_lang_data in enumerate(dataloader):
        if it >= 20:
            break
        _, _, idx = train_lang_data
        curr_traj_embed = traj_embeds[idx]
        curr_traj_reward = torch.sum(torch.from_numpy(true_reward) * curr_feature_value.numpy())
        sampled_traj_true_rewards.append(curr_traj_reward.view(1, -1))

        print(f"\n...Query {it + 1}...")

        # get the language feedback from the user
        curr_traj_images = traj_img_obs[idx]
        print("Replaying the current trajectory...\n")
        replay_trajectory_video(curr_traj_images, title='Current Trajectory', frame_rate=20)

        nlcomp = input("Please provide the language feedback: ")
        nlcomp = remove_special_characters(nlcomp)
        lang_embed = get_lang_embed(nlcomp, model, device, tokenizer, lang_model=lang_encoder)

        all_lang_feedback.append(nlcomp)
        all_lang_embeds.append(lang_embed)

        # Get the feature of the language comparison
        nlcomp_features = torch.tensor(np.array(all_lang_embeds))

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

        for i in range(args.num_iterations):
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

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
                
        if (it + 1) % 5 == 0:
            if it >= 15:
                # Show users the best trajectory so far and let them rate it
                optim_traj_idx = get_optimal_traj(learned_reward, optim_candidates_embeds)

                # print(f"Optimal Trajectory Index: {optim_traj_idx}")
                if args.real_robot:
                    print("\nReplaying the optimal trajectory on the robot...")
                    replay_traj_widowx(widowx_env, optim_candidates_policy_outs[optim_traj_idx])
                else:
                    print("\nReplaying the optimal trajectory...")
                    
                    replay_trajectory_video(optim_candidates_images[optim_traj_idx], title='Optimal Trajectory', frame_rate=10)
                score = input("Please rate the trajectory (0-5, 5 is the best): ")
                optim_traj_scores.append(int(score))
            
            else:
                optim_traj_idx = get_optimal_traj(learned_reward, traj_embeds)

                print("\nReplaying the optimal trajectory...")
                replay_trajectory_video(traj_img_obs[optim_traj_idx], title='Optimal Trajectory', frame_rate=10)
                
                score = input("Please rate the trajectory (0-5, 5 is the best): ")
                optim_traj_scores.append(int(score))
    
    return optim_traj_scores


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
    # Load data
    
    data_dict = load_data(args)
    trajs = data_dict["trajs"]
    traj_img_obs = data_dict["traj_img_obs"]
    nlcomps_embed = lang_data_dict["nlcomp_embeds"]

    feature_values = np.array([get_feature_value(traj) for traj in trajs])

    feature_value_means = np.mean(feature_values, axis=0)
    feature_value_stds = np.std(feature_values, axis=0)

    # Normalize the feature values
    feature_values = (feature_values - feature_value_means) / feature_value_stds

    # Initialize the dataset and dataloader
    # lang_dataset = LangPrefDataset(trajs, feature_values)
    # lang_data = DataLoader(lang_dataset, batch_size=1, shuffle=True)

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

    if args.env == "widowx":
        STATE_OBS_DIM = WidowX_STATE_OBS_DIM
        ACTION_DIM = WidowX_ACTION_DIM
        PROPRIO_STATE_DIM = WidowX_PROPRIO_STATE_DIM
        OBJECT_STATE_DIM = WidowX_OBJECT_STATE_DIM
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
    if args.old_model:
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k.replace("_hidden_layer", ".0")
            new_k = new_k.replace("_output_layer", ".2")
            new_state_dict[new_k] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Check if the embeddings are already computed
    # If not, compute the embeddings
    if not os.path.exists(f"{args.data_dir}/traj_embeds.npy"):
        traj_embeds, lang_embeds = get_traj_lang_embeds(
            trajs,
            train_nlcomps,
            model,
            device,
            args.use_bert_encoder,
            tokenizer,
            nlcomps_bert_embeds=train_nlcomps_embed,
            use_img_obs=args.use_img_obs,
            img_obs=train_img_obs,         
        )

        # Save the embeddings
        np.save(f"{args.data_dir}/traj_embeds.npy", traj_embeds)
        np.save(f"{args.data_dir}/lang_embeds.npy", lang_embeds)
    else:
        traj_embeds = np.load(f"{args.data_dir}/traj_embeds.npy")
        lang_embeds = np.load(f"{args.data_dir}/lang_embeds.npy")

    print("Mean Norm of Traj Embeds:", np.linalg.norm(traj_embeds, axis=1).mean())
    print("Norm of feature values:", np.linalg.norm(feature_values, axis=1).mean())

    # Random init learned reward
    learned_reward = RewardFunc(feature_dim, 1)
    optimizer = torch.optim.SGD(learned_reward.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.95)

    learned_reward_noiseless = RewardFunc(feature_dim, 1)
    # Copy the weights from the noisy reward
    learned_reward_noiseless.linear.weight.data = learned_reward.linear.weight.data.clone()
    optimizer_noiseless = torch.optim.SGD(
        learned_reward_noiseless.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    lang_pref_learning(
        args,
        trajs,
        feature_values
        model,
        None,
        None,
        None,
        None,
        learned_reward,
        traj_embeds,
        traj_img_obs,
        lang_embeds,
        optimizer,
        device,
        tokenizer,
        lang_encoder,
    )
    
    else:
        raise ValueError("Invalid method")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--env", type=str, default="widowx", help="")
    parser.add_argument("--data-dir", type=str, default="data", help="")
    parser.add_argument("--model-dir", type=str, default="models", help="")
    parser.add_argument("--old-model", action="store_true", help="")

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
        "--method",
        default="lang",
        type=str,
        choices=["lang", "comp"],
    )
    parser.add_argument(
        "--real-robot",
        action="store_true",
        help="whether to run on real robot",
    )
    parser.add_argument(
        "--active",
        default=4,
        type=int,
        choices=[1, 2, 3, 4],
        help="Choice for Active Learning method",
    )
    parser.add_argument(
        "--lang",
        default=1,
        type=int,
        choices=[1, 2, 3, 4],
        help="Choice for Language sampling method",
    )
    parser.add_argument(
        "--reward",
        default=1,
        type=int,
        choices=[1, 2, 3, 4],
        help="Choice for Reward sampling method",
    )

    args = parser.parse_args()
    run(args)