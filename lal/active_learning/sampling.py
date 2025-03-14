import torch
import numpy as np
from lal.active_learning.mh import mh_w, mh_l
from lal.active_learning.gibbs import gibbs_w, gibbs_l
from lal.active_learning.laplace import laplace_w, laplace_l, laplace_l_sampling
from lal.active_learning.ep import ep_w, ep_l, ep_w_dimension
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
		self.dim = dim # 512 for t5-small, 768 for

		self.reward = reward
		self.w_sampler = None
		if reward == 1: self.w_sampler = mh_w
		elif reward == 2: self.w_sampler = gibbs_w
		elif reward == 3: self.w_sampler = laplace_w
		elif reward == 4: self.w_sampler = ep_w_dimension
		assert self.w_sampler is not None
		
		self.lang = lang
		self.l_sampler = None
		if lang == 1: self.l_sampler = mh_l
		elif lang == 2: self.l_sampler = gibbs_l
		elif lang == 3: self.l_sampler = laplace_l
		elif lang == 4: self.l_sampler = ep_l
		assert self.l_sampler is not None

		if reward == 3 and lang == 3: self.l_sampler = laplace_l_sampling # was doing this b4, but this isn't properly finding l's per w. it finds the l mode for all the w's
	
	# def sample(self, queries, initial_w=None, num_w_samples=200, burn_in_w=2000, thin_w=200, num_l_samples_per_w=100, burn_in_l=100, thin_l=5, seed=0): # 1-1
	# def sample(self, queries, initial_w=None, num_w_samples=100, burn_in_w=1000, thin_w=50, num_l_samples_per_w=100, burn_in_l=100, thin_l=5, seed=0): # 2-1
	# def sample(self, queries, initial_w=None, num_w_samples=500, burn_in_w=100, thin_w=5, num_l_samples_per_w=1, burn_in_l=0, thin_l=1, seed=0): # 3-1 # used in metaworld
	# def sample(self, queries, initial_w=None, num_w_samples=100, burn_in_w=100, thin_w=5, num_l_samples_per_w=1, burn_in_l=0, thin_l=1, seed=0): # 3-1 # testing for robosuite
	def sample(self, queries, initial_w=None, num_w_samples=10, burn_in_w=0, thin_w=1, num_l_samples_per_w=5, burn_in_l=0, thin_l=1, seed=0): # 3-3 with laplace_l_sampling for robosuite
	# def sample(self, queries, initial_w=None, num_w_samples=50, burn_in_w=0, thin_w=1, num_l_samples_per_w=100, burn_in_l=0, thin_l=1, seed=0): # 3-3 with laplace_l
	# def sample(self, queries, initial_w=None, num_w_samples=100, burn_in_w=100, thin_w=5, num_l_samples_per_w=100, burn_in_l=100, thin_l=5, seed=0): # 4-1
		'''
		Using parallelization, sample from reward weight space
		parameters:
        	queries (type list): the list of all query feedbacks provided by the user
			initial_w (type torch.tensor): the mode of the previous time
			num_w_samples (type int): The number of w to sample
			num_l_samples (type int): The number of l to sample
			burn_in (type int): the number of samples to burn
        	thin (type int): the increment that we take from the samples to thin out the set
		returns:
			w_samples (type np.array): The sampled reward weight vectors, size 512 for t5-small
			l_samples (type np.array): The sampled language embeddings, size 512 for t5-small
		'''
		w_samples = []
		l_samples = []

		# traj_embeds = np.zeros((1, 512))
		# feedback_embeds = np.zeros((1, 512))
		# if len(queries) > 1:
		# 	traj_embeds = []
		# 	feedback_embeds = []
		# 	for i in range(1, len(queries)):
		# 		traj_embeds.append(queries[i][0])
		# 		feedback_embeds.append(queries[i][1])
		# 	traj_embeds = np.stack(traj_embeds)
		# 	feedback_embeds = np.stack(feedback_embeds)
		traj_embeds = [np.zeros(self.dim)]
		feedback_embeds = [np.zeros(self.dim)]
		for i in range(0, len(queries)):
			traj_embeds.append(queries[i][0].reshape(-1))
			feedback_embeds.append(queries[i][1].reshape(-1))
		# for i in range(0, len(traj_embeds)):
		# 	print(traj_embeds[i].shape)
		traj_embeds = np.stack(traj_embeds)
		feedback_embeds = np.stack(feedback_embeds)
		if self.reward <= 2 and self.lang <= 2: # MC-MC
			prev_w = initial_w
			for i in range(num_w_samples * thin_w + burn_in_w):
				w = self.w_sampler(traj_embeds, feedback_embeds, self.dim, prev_w, seed=seed) # do one step in mcmc to sample w
				prev_w = w
				if i >= burn_in_w and i % thin_w == 0:
					w_samples.append(w)
					l_samples.append(self.l_sampler(traj_embeds, w, self.dim, num_l_samples_per_w * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l))

		elif self.reward <= 2: # MC-Sampling
			prev_w = initial_w # might need to parallelize this one
			for i in range(num_w_samples * thin_w + burn_in_w):
				w = self.w_sampler(traj_embeds, feedback_embeds, self.dim, prev_w) # do one step in mcmc to sample w
				prev_w = w
				if i >= burn_in_w and i % thin_w == 0:
					w_samples.append(w)
					l_samples.append(self.l_sampler(traj_embeds, w, self.dim, num_l_samples_per_w))

		elif self.reward >= 3 and self.lang <= 2: # Sampling-MC
			w_samples = self.w_sampler(traj_embeds, feedback_embeds, self.dim, num_w_samples).astype(np.float32) #  approx the distribution into gaussian, and sample from using torch
			# with ProcessPoolExecutor() as executor: # parallelize the l sampling (if needed)
			for i in range(num_w_samples):
				# future = executor.submit(self.l_sampler, traj_embeds, w_samples[i], self.dim, num_l_samples_per_w * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l)
				# l_samples.append(future.result())
				l_samples.append(self.l_sampler(traj_embeds, w_samples[i], self.dim, num_l_samples_per_w * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l, seed=seed))
		else:
			# start_time = time.time()
			w_samples = self.w_sampler(traj_embeds, feedback_embeds, self.dim, num_w_samples) #  approx the distribution into gaussian, and sample from using torch
			# print("w sampling: ", time.time() - start_time)
			# start_time = time.time()
			
			# l_samples = self.l_sampler(traj_embeds, w_samples, self.dim, num_w_samples, num_l_samples_per_w)
			l_samples = self.l_sampler(traj_embeds, np.mean(w_samples, axis=0).reshape(1, -1), self.dim, num_w_samples, num_l_samples_per_w)
			
			# print("l sampling: ", time.time() - start_time)
			# for i in range(num_w_samples):
				# l_samples.append(self.l_sampler(traj_embeds, w_samples[i], self.dim, num_l_samples_per_w))
			l_samples = np.stack(l_samples)
		
		if self.reward <= 2: w_samples = torch.from_numpy(np.stack(w_samples))
		else: w_samples = torch.from_numpy(w_samples)
		if self.lang <= 2: l_samples = torch.from_numpy(np.stack(l_samples))
		else: l_samples = torch.from_numpy(l_samples)
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

		self.reward = reward
		self.w_sampler = None
		if reward == 1: self.w_sampler = mh_w
		elif reward == 2: self.w_sampler = gibbs_w
		elif reward == 3: self.w_sampler = laplace_w
		elif reward == 4: self.w_sampler = ep_w_dimension
		assert self.w_sampler is not None

	# def sample(self, queries, initial_w=None, num_w_samples=100, burn_in_w=100000, thin_w=100, seed=0): # 1
	def sample(self, queries, initial_w=None, num_w_samples=100, burn_in_w=1000, thin_w=50, seed=0): # 2
	# def sample(self, queries, initial_w=None, num_w_samples=100, burn_in_w=1000, thin_w=50, seed=0): # 3 and 4
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

		traj_embeds = [np.zeros(self.dim)]
		feedback_embeds = [np.zeros(self.dim)]
		for i in range(0, len(queries)):
			traj_embeds.append(queries[i][0].reshape(-1))
			feedback_embeds.append(queries[i][1].reshape(-1))
		traj_embeds = np.stack(traj_embeds)
		feedback_embeds = np.stack(feedback_embeds)

		if self.reward <= 2: # MC
			# prev_w = initial_w
			prev_w = np.zeros(self.dim, dtype=np.float32)
			for i in range(num_w_samples * thin_w + burn_in_w):
				w = self.w_sampler(traj_embeds, feedback_embeds, self.dim, prev_w, seed=seed) # do one step in mcmc to sample w
				prev_w = w
				if i >= burn_in_w and i % thin_w == 0:
					w_samples.append(w)

		elif self.reward >= 3: # Sampling-MC
			w_samples = self.w_sampler(traj_embeds, feedback_embeds, self.dim, num_w_samples).astype(np.float32) #  approx the distribution into gaussian, and sample from using torch
		
		if self.reward <= 2: w_samples = torch.from_numpy(np.stack(w_samples))
		else: w_samples = torch.from_numpy(w_samples)
		return w_samples

class LanguageSampler:
	'''
	This is the sampler for only language. Only samples l given the reward weights.
	'''
	def __init__(self, dim, lang=0):
		'''
		parameters:
			dim (type int): The size embedding space
			reward (type int): Specifies which sampling method will be used for sampling from reward weight space
			lang (type int): Specifies which sampling method will be used for sampling from language space
		'''
		self.dim = dim
		
		self.lang = lang
		self.l_sampler = None
		if lang == 1: self.l_sampler = mh_l
		elif lang == 2: self.l_sampler = gibbs_l
		elif lang == 3: self.l_sampler = laplace_l_sampling
		elif lang == 4: self.l_sampler = ep_l
		assert self.l_sampler is not None
	
	def sample(self, queries, w_samples=None, num_l_samples=1, burn_in_l=0, thin_l=1, seed=0):
		'''
		Using parallelization, sample from reward weight space
		parameters:
        	queries (type list): the list of all query feedbacks provided by the user
			initial_w (type torch.tensor): the mode of the previous time
			num_w_samples (type int): The number of w to sample
			num_l_samples (type int): The number of l to sample
			burn_in (type int): the number of samples to burn
        	thin (type int): the increment that we take from the samples to thin out the set
		returns:
			w_samples (type np.array): The sampled reward weight vectors, size 512 for t5-small
			l_samples (type np.array): The sampled language embeddings, size 512 for t5-small
		'''
		l_samples = []

		traj_embeds = [np.zeros(self.dim)]
		feedback_embeds = [np.zeros(self.dim)]
		for i in range(0, len(queries)):
			traj_embeds.append(queries[i][0].reshape(-1))
			feedback_embeds.append(queries[i][1].reshape(-1))
		traj_embeds = np.stack(traj_embeds)
		feedback_embeds = np.stack(feedback_embeds)

		if self.lang <= 2: # MC-MC
			for i in range(len(w_samples)):
				l_samples.append(self.l_sampler(traj_embeds, w_samples[i], self.dim, num_l_samples * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l, seed=seed))
			l_samples = torch.tensor(np.array(l_samples))
		else:
			l_samples = torch.from_numpy(self.l_sampler(traj_embeds, w_samples, self.dim, len(w_samples), num_l_samples))
		return l_samples