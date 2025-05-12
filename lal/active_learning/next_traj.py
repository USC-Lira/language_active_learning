import torch
import numpy as np
from lal.active_learning.sampling import EmbeddingSampler, WeightSampler, LanguageSampler
from lal.active_learning.info import info
import time
import copy

def add_noise_to_weights(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            m.weight.add_(torch.randn(m.weight.size()) * 0.2)

def next_traj_method(traj_embeds=None, lang_embeds=None, initial_w=None, active=0, reward=1, lang=1, seed=0):
    '''
    From the correct active learning method and sampling methods, return the correct iter class
    '''
    assert active != 0
    next_traj_iter = None

    # sampling from embedding space and reward space
    if active == 1:
        assert traj_embeds is not None
        next_traj_iter = OpenIter(traj_embeds, lang_embeds, reward, lang, initial_w, seed)

    # BALD
    elif active == 5:
        assert traj_embeds is not None and lang_embeds is not None
        next_traj_iter = BALDIter(traj_embeds, lang_embeds, initial_w, lang, seed)

    return next_traj_iter

class OpenIter:
    '''
    Object that returns the next trajectory. Using information gain, select the next trajectory.
    Sample from reward space, then based on each reward space, sample from open vocabulary embedding space.
    '''
    def __init__(self, traj_embeds, lang_embeds, reward, lang, initial_w, seed):
        '''
        parameters:
            traj_embeds (type np.array): Contains the trajectory embedding, a float vector size 512
            lang_embeds (type np.array): Contains the language embedding, a float vector size 512
            reward (type int): Reward space sampling method
        '''
        self.traj_embeds = traj_embeds
        self.lang_embeds = lang_embeds
        self.dim = self.traj_embeds[-1].shape[0]
        self.sampler = EmbeddingSampler(self.dim, reward=reward, lang=lang)
        self.queries = []
        self.initial_w = initial_w
        self.seed = seed
        self.prev_idxs = []

    def feed(self, curr_traj_embed, feedback_embed, pos):
        '''
        Every time a human feedback is given from a query, add the results to our list
        parameters:
        	curr_traj_embed (type np.array): Embedding of the current trajectory
            feedback_embed (type np.array): Embedding of the feedback given
            pos (type bool): Whether or not feedback is positive or negative
        '''
        self.queries.append([curr_traj_embed.numpy(), feedback_embed.numpy(), pos])

    def next(self):
        '''
        Sample w and l, perform info gain to get next trajectory idx
        returns:
			w_samples (type np.array): The sampled reward weight vectors, size 512 for t5-small
            idx (type int): An int that contains the idx of the trajectory we want to show
            ig (type float): The amount of info gained
        '''
        guess = np.zeros(self.dim)
        w_samples, l_samples = self.sampler.sample(self.queries, initial_w=guess, seed=self.seed)
        idx, ig = info(w_samples, l_samples, self.traj_embeds, self.prev_idxs)
        self.prev_idxs.append(idx)
        self.prev_idxs.sort()
        return w_samples, idx, ig

class BALDIter:
    '''
    Object that returns the next trajectory. Using information gain, select the next trajectory.
    Sample from reward space, then based on each reward space, sample from open vocabulary embedding space.
    '''
    def __init__(self, traj_embeds, lang_embeds, models, lang, seed):
        '''
        parameters:
            traj_embeds (type np.array): Contains the trajectory embedding, a float vector size 512
            lang_embeds (type np.array): Contains the language embedding, a float vector size 512
            models (type torch.tensor): Neural network reward models
            seed (type int): Random seed
        '''
        self.traj_embeds = traj_embeds
        self.dim = self.traj_embeds[-1].shape[0]
        self.sampler = LanguageSampler(self.dim, lang=lang)
        self.queries = []
        self.models = models
        self.seed = seed
        self.prev_idxs = []

    def feed(self, curr_traj_embed, feedback_embed, pos):
        '''
        Every time a human feedback is given from a query, add the results to our list
        parameters:
        	curr_traj_embed (type np.array): Embedding of the current trajectory
            feedback_embed (type np.array): Embedding of the feedback given
            pos (type bool): Whether or not feedback is positive or negative
        '''
        self.queries.append([curr_traj_embed.numpy(), feedback_embed.numpy(), pos])

    def next(self):
        '''
        Sample w and l, perform info gain to get next trajectory idx
        returns:
			w_samples (type np.array): The sampled reward weight vectors, size 512 for t5-small
            idx (type int): An int that contains the idx of the trajectory we want to show
            ig (type float): The amount of info gained
        '''
        noisy_models = copy.deepcopy(self.models)
        for noisy_model in noisy_models:
            noisy_model.apply(add_noise_to_weights)
        w_samples = np.stack([model.linear.weight.detach().numpy() for model in noisy_models]).squeeze(1)

        l_samples = self.sampler.sample(self.queries, w_samples, seed=self.seed)
        idx, ig = info(torch.tensor(w_samples).to(torch.double), l_samples, self.traj_embeds, self.prev_idxs)
        self.prev_idxs.append(idx)
        self.prev_idxs.sort()
        return idx, ig