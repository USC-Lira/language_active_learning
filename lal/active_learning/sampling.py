import torch
import numpy as np
from lal.active_learning.mh import mh_w, mh_l
from lal.active_learning.gibbs import gibbs_w, gibbs_l
from lal.active_learning.laplace import laplace_w, laplace_l
from lal.active_learning.ep import ep_w, ep_l
import time
from concurrent.futures import ProcessPoolExecutor

class EmbeddingSampler:
	'''
	This is the sampler for both reward weights and language. It is only used for sampling from embedding space, and samples from both weights and embeddings in parallel to reduce cost
	'''
	def __init__(self, dim, reward=0, lang=0):
		'''
		parameters:
			dim (type int): The size embedding space
			reward (type int): Specifies which sampling method will be used for sampling from reward weight space
			lang (type int): Specifies which sampling method will be used for sampling from language space
		'''
		self.dim = dim
		self.feedback_embeddings = []

		self.reward = reward
		self.w_sampler = None
		if reward == 1: self.w_sampler = mh_w
		elif reward == 2: self.w_sampler = gibbs_w
		elif reward == 3: self.w_sampler = laplace_w
		elif reward == 4: self.w_sampler = ep_w
		assert self.w_sampler is not None
		
		self.lang = lang
		self.l_sampler = None
		if lang == 1: self.l_sampler = mh_l
		elif lang == 2: self.l_sampler = gibbs_l
		elif lang == 3: self.l_sampler = laplace_l
		elif lang == 4: self.l_sampler = ep_l
		assert self.l_sampler is not None
	
	def sample(self, queries, initial_w=None, num_w_samples=100, burn_in_w=1000, thin_w=50, num_l_samples_per_w=10, burn_in_l=100, thin_l=5, seed=0):
	# def sample(self, queries, initial_w=None, num_w_samples=10, burn_in_w=10, thin_w=10, num_l_samples_per_w=5, burn_in_l=10, thin_l=10, seed=0): # for bug-testing purposes
		'''
		Using parallelization, sample from reward weight space
		parameters:
        	queries (type list): the list of all query feedbacks provided by the user
			initial_w (type torch.tensor): the mode of the previous time
			num_w_samples (type int): The number of w to sample
			num_l_samples (type int): The number of l to sample
			burn_in (type int): the number of samples to burn
        	thin (type int): the increment that we take from the samples to thin out the set
        	seed (type int): the random seed
		returns:
			w_samples (type np.array): The sampled reward weight vectors, size 512 for t5-small
			l_samples (type np.array): The sampled language embeddings, size 512 for t5-small
		'''
		w_samples = []
		l_samples = []

		if self.reward <= 2 and self.lang == 1: # MC-MH
			prev_w = initial_w
			for i in range(num_w_samples * thin_w + burn_in_w):
				w = self.w_sampler(queries, self.dim, prev_w, seed=seed) # do one step in mcmc to sample w
				prev_w = w
				if i >= burn_in_w and i % thin_w == 0:
					w_samples.append(w)
					l_samples.append(self.l_sampler(queries, w, self.dim, num_l_samples_per_w * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l, seed=seed + i))
		
		elif self.reward <= 2 and self.lang == 2: # MC-Gibbs
			with ProcessPoolExecutor() as executor: # parallelize the l sampling
				prev_w = initial_w # might need to parallelize this one
				for i in range(num_w_samples * thin_w + burn_in_w):
					w = self.w_sampler(queries, self.dim, prev_w, seed=seed) # do one step in mcmc to sample w
					prev_w = w
					if i >= burn_in_w and i % thin_w == 0:
						w_samples.append(w)
						future = executor.submit(self.l_sampler, queries, w, self.dim, num_l_samples_per_w * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l, seed=seed + i)
						l_samples.append(future.result())

		elif self.reward <= 2: # MC-Sampling
			prev_w = initial_w # might need to parallelize this one
			for i in range(num_w_samples * thin_w + burn_in_w):
				w = self.w_sampler(queries, self.dim, prev_w, seed=seed) # do one step in mcmc to sample w
				prev_w = w
				if i >= burn_in_w and i % thin_w == 0:
					w_samples.append(w)
					l_samples.append(self.l_sampler(queries, w, self.dim, num_l_samples_per_w, seed=seed + i))

		elif self.reward >= 3 and self.lang <= 2: # Sampling-MC
			w_samples = self.w_sampler(queries, self.dim, num_w_samples, seed=seed) #  approx the distribution into gaussian, and sample from using torch
			with ProcessPoolExecutor() as executor: # parallelize the l sampling (if needed)
				for i in range(num_w_samples):
					future = executor.submit(self.l_sampler, queries, w_samples[i], self.dim, num_l_samples_per_w * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l, seed=seed + i)
					l_samples.append(future.result())

		else: # Sampling-Sampling
			w_samples = self.w_sampler(queries, self.dim, num_w_samples, seed=seed) #  approx the distribution into gaussian, and sample from using torch
			with ProcessPoolExecutor() as executor: # parallelize the l sampling (if needed)
				for i in range(num_w_samples):
					future = executor.submit(self.l_sampler, queries, w_samples[i], self.dim, num_l_samples_per_w, seed=seed + i)
					l_samples.append(future.result())
		
		if self.reward <= 2: w_samples = torch.stack(w_samples)
		l_samples = torch.stack(l_samples)
		return w_samples, l_samples

class WeightSampler:
	'''
	This is the reward weight sampler. It is used for all active learning methods besides sampling from embedding space; this is only used to sample from w space.
	'''
	def __init__(self, dim, reward=0):
		'''
		parameters:
			dim (type int): The size embedding space
			reward (type int): Specifies which sampling method will be used for sampling from reward weight space
		'''
		self.dim = dim
		self.feedback_embeddings = []

		self.reward = reward
		self.w_sampler = None
		if reward == 1: self.w_sampler = mh_w
		elif reward == 2: self.w_sampler = gibbs_w
		elif reward == 3: self.w_sampler = laplace_w
		elif reward == 4: self.w_sampler = ep_w
		assert self.w_sampler is not None

	def sample(self, queries, initial_w=None, num_w_samples=100, num_l_samples_per_w=None, burn_in=50, thin=50, seed=0):
		'''
		Sample from reward weight space. No parallelization is needed (albeit it's possible for laplace and ep after estimating the distribution)
		parameters:
			queries (type list): the list of all query feedbacks provided by the user
			num_w_samples (type int): The number of w to sample
			num_l_samples (type None): Here to match input as EmbeddingSampler
			burn_in (type int): the number of samples to burn
        	thin (type int): the increment that we take from the samples to thin out the set
        	seed (type int): the random seed
		returns:
			w_samples (type np.array): The sampled reward weight vectors, size 512 for t5-small
			np.array([]) (type np.array): Return an empty list to have the same return criteria as BothSampler()
		'''
		w_samples = []
	
		if self.reward >= 3: # w is sampled using approximation
			w_samples = self.w_sampler(queries, self.dim, num_w_samples, seed=seed) #  approx the distribution into gaussian, and sample from using torch
		elif self.reward <= 2: # w is sampled using mcmc
			prev_w = initial_w
			for i in range(num_w_samples):
				w = self.w_sampler(queries, self.dim, prev_w, burn_in=burn_in, thin=thin, seed=seed) # do one step in mcmc to sample w
				prev_w = w
				if (i + burn) % thin == 0:
					w_samples.append(w)

		if self.reward >= 3:
			w_samples = self.w_sampler(num_w_samples, seed, self.queries)
		elif self.reward <= 2:
			w_samples = self.w_sampler(num_w_samples, seed, self.queries, self.initial)
		return w_samples, np.array([])