import torch
import numpy as np
from lal.active_learning.sampling import EmbeddingSampler, WeightSampler, LanguageSampler
from lal.active_learning.info import info
import time

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

    # sample from llm and reward space
    elif active == 2:
        assert traj_embeds is not None
        next_traj_iter = LLMIter(traj_embeds, None, reward)

    # sample from dataset and reward space
    elif active == 3:
        assert traj_embeds is not None and lang_embeds is not None
        next_traj_iter = DatasetIter(traj_embeds, lang_embeds, reward)
    
    # sample completely randomly
    elif active == 4:
        assert traj_embeds is not None
        next_traj_iter = RandomIter(traj_embeds, reward, initial_w, seed)

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
        # self.w_M = 10
        # self.l_M_start = 100
        # self.l_M_end = 20
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
        # start_time = time.time()
        w_samples, l_samples = self.sampler.sample(self.queries, initial_w=self.initial_w.numpy(), seed=self.seed)
        # print(f"Sampling took: {time.time() - start_time} seconds")
        # self.initial_w = w_samples[-1]
        self.initial_w = torch.mean(w_samples, dim=0)
        # start_time = time.time()
        idx, ig = info(w_samples, l_samples, self.traj_embeds, self.prev_idxs)
        # print(f"Info gain took: {time.time() - start_time} seconds")
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
        self.w_samples = np.stack([model.linear.weight.detach().numpy() for model in models]).squeeze(1)
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
        l_samples = self.sampler.sample(self.queries, self.w_samples, seed=self.seed)
        idx, ig = info(torch.tensor(self.w_samples).to(torch.double), l_samples, self.traj_embeds, self.prev_idxs)
        self.prev_idxs.append(idx)
        self.prev_idxs.sort()
        # print(f"Info gain took: {time.time() - start_time} seconds")
        return idx, ig

class LLMIter:
    '''
    Object that returns the next trajectory. Using information gain, select the next trajectory.
    Sample language from an LLM, then assign weights based on our reward models. Finally, resample after newly assigned weights.
    '''
    def __init__(self, traj_embeds, lang_embeds, reward, initial_w):
        '''
        parameters:
            traj_embeds (type np.array): Contains the trajectory embedding, a float vector size 512
            lang_embeds (type np.array): Contains the language embedding, a float vector size 512
            reward (type int): Reward space sampling method
        '''
        self.traj_embeds = traj_embeds
        self.lang_embeds = lang_embeds
        self.dim = self.traj_embeds[-1].shape[0]
        self.sampler = WeightSampler(dim, reward=reward)
        self.queries = []
        self.initial_w = initial_w

    def feed(self, query):
        '''
        Every time a human feedback is given from a query, add the results to our list
        parameters:
        	queries (type list): the list of all query feedbacks provided by the user
        '''
        self.queries.append(query)

    def next(self):
        '''
        Sample w and l, perform info gain to get next trajectory idx
        returns:
			w_samples (type np.array): The sampled reward weight vectors, size 512 for t5-small
            idx (type int): An int that contains the idx of the trajectory we want to show
        '''
        w_samples = self.sampler.sample(self.queries, initial_w=self.initial_w.numpy(), seed=self.seed)
        # self.initial_w = w_samples[-1]
        self.initial_w = torch.mean(w_samples, dim=0)
        # l_samples = query LLM here
        idx, ig = info(w_samples, l_samples, self.traj_embeds)
        return w_samples, idx, ig

class DatasetIter:
    '''
    Object that returns the next trajectory. Using information gain, select the next trajectory.
    Uniformly sample language from the dataset, then assign weights based on our reward models. Finally, resample after newly assigned weights.
    '''
    def __init__(self, traj_embeds, lang_embeds, reward, initial_w):
        '''
        parameters:
            traj_embeds (type np.array): Contains the trajectory embedding, a float vector size 512
            lang_embeds (type np.array): Contains the language embedding, a float vector size 512
            reward (type int): Reward space sampling method
        '''
        self.traj_embeds = traj_embeds
        self.lang_embeds = lang_embeds
        self.dim = self.traj_embeds[-1].shape[0]
        self.sampler = WeightSampler(self.dim, reward=reward)
        self.l_M_start = 100
        self.l_M_end = 10
        self.queries = []
        self.initial_w = initial_w

    def feed(self, curr_traj_embed, feedback_embed, pos):
        '''
        Every time a human feedback is given from a query, add the results to our list
        parameters:
        	queries (type list): the list of all query feedbacks provided by the user
        '''
        self.queries.append([curr_traj_embed, feedback_embed, pos])

    def next(self):
        '''
        Sample w and l, perform info gain to get next trajectory idx
        returns:
			w_samples (type np.array): The sampled reward weight vectors, size 512 for t5-small
            idx (type int): An int that contains the idx of the trajectory we want to show
        '''
        w_samples = self.sampler.sample(self.queries, initial_w=self.initial_w.numpy(), seed=self.seed)
        # self.initial_w = w_samples[-1]
        self.initial_w = torch.mean(w_samples, dim=0)
        l_samples = []
        for i in range(len(w_samples)):
            curr_w = w_samples[i]

            l_idxs = np.random.choice(len(self.lang_embeds), size=self.l_M_start, replace=False)
            pre_l_samples = self.lang_embeds[l_idxs]

            unnormalized_p = pre_l_samples @ curr_w # is this fine? or should we be doing l(w-t); maybe this is fine
            normalized_p = pre_l_samples / torch.norm(pre_l_samples)

            new_l_idxs = np.random.choice(l_idxs, size=self.l_M_end, replace=False, p=normalized_p)
            post_l_samples = self.lang_embeds[new_l_idxs]
            l_samples.append(post_l_samples)
        l_samples = np.stack(l_samples)
        idx, ig = info(w_samples, l_samples, self.traj_embeds)
        return w_samples, idx, ig

class RandomIter:
    '''
    Object that returns the next trajectory. Randomly get the next trajectory w/o respect to the reward model
    '''
    def __init__(self, traj_embeds, reward, initial_w, seed):
        '''
        parameters:
            traj_embeds (type np.array): Contains the trajectory embedding, a float vector size 512
        '''
        self.traj_embeds = traj_embeds
        self.dim = self.traj_embeds[-1].shape[0]
        self.sampler = WeightSampler(self.dim, reward=reward)
        self.queries = []
        self.initial_w = initial_w
        self.seed = seed
        self.dataset = np.random.choice(len(self.traj_embeds), size=len(self.traj_embeds), replace=False)
        self.counter = 0

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
        Sample w, randomly select next trajectory idx
        returns:
			w_samples (type np.array): The sampled reward weight vectors, size 512 for t5-small
            idx (type int): An int that contains the idx of the trajectory we want to show
            np.inf (type int): Placeholder for info gain score
        '''
        w_samples = self.sampler.sample(self.queries, initial_w=self.initial_w.numpy(), seed=self.seed)
        # self.initial_w = w_samples[-1]
        self.initial_w = torch.mean(w_samples, dim=0)
        idx = self.dataset[self.counter]
        self.counter += 1
        return w_samples, idx, np.inf