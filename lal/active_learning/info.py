import torch
import numpy as np

def info(w_samples, l_samples, traj_embeds, prev_idxs):
	'''
	Batched matrix algebra for information gain equation
	parameters:
		w_samples (type torch.tensor): sampled list of reward weights
		l_samples (type torch.tensor): sampled list of language embeddings
		traj_embeds (type np.array): the dataset of trajectory embeddings
	returns:
		idx (type int): the index of the next trajectory to query the human that provides the best info gain
		ig (type float): the max amount of info gained
	'''
	modified_traj_embeds = np.delete(traj_embeds, prev_idxs, 0)
	M = w_samples.shape[0] # w/ shape (M, dim)
	dim = w_samples.shape[1]
	K = l_samples.shape[1] # w/ shape (M, K, dim)
	T = modified_traj_embeds.shape[0] # (T, dim)

	embed_diff = (w_samples.unsqueeze(1) - modified_traj_embeds.unsqueeze(0)).reshape(M*T, dim) # shape (M*T, dim)
	align = np.exp(embed_diff @ l_samples.reshape(M*K, dim).T) # shape (M*T, M*K)
	probs = align.reshape(M, T, -1).transpose(0, 1) # shape (T, M, M*K), so: (trajectory, weight, language)
	probs /= torch.sum(probs, dim=-1).unsqueeze(-1) # shape (T, M, M*K) # this step is here b/c the prob fxn is actually equal to the numerator divide by all possible language
	tmp = M * probs / torch.sum(probs, dim=1).unsqueeze(1)
	f_values = torch.sum(torch.sum(torch.log2(tmp), dim=2), dim=1) / (M*K) # shape (T)
	
	idx = torch.argmax(f_values) # easy code uses argmin but also divides by negative M for some reason
	ig = torch.abs(torch.nan_to_num(f_values[idx], nan=1))

	for i in prev_idxs:
		if idx >= i: idx += 1
		else: break
	return idx, ig


if __name__ == "__main__":
	# info gain is more efficient when there are more l
	w_samples = torch.randn(100, 512)
	w_samples /= torch.norm(w_samples, dim=1).reshape(100, 1)
	l_samples = torch.randn(100, 100, 512)
	l_samples /= torch.norm(l_samples, dim=2).reshape(100, 100, 1)
	traj_embeds = torch.randn(1000, 512)
	traj_embeds /= torch.norm(traj_embeds, dim=1).reshape(1000, 1)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	w_samples.to(device)
	l_samples.to(device)
	traj_embeds.to(device)
	import time
	start_time = time.time()
	idx, ig = info(w_samples, l_samples, traj_embeds)
	print(f"Time: {time.time() - start_time}")