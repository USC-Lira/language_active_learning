import torch
import numpy as np

def info(w_samples, l_samples, traj_embeds):
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
	M = w_samples.shape[0] # w/ shape (M, dim)
	dim = w_samples.shape[1]
	K = l_samples.shape[1] # w/ shape (M, K, dim)
	T = traj_embeds.shape[0] # (T, dim)

	# embed_diff = (w_samples.unsqueeze(1) - traj_embeds.unsqueeze(0)).reshape(M, T, dim) # shape (M, T, dim)
	# align = torch.einsum('mtd,mqd->mtq', embed_diff, l_samples) # shape (M, T, K) # it shouldn't be (M, T, K) bc this only allows l's to multiply w/ the w it was sampled from, but not w/ other w's
	# numerator = torch.exp(torch.transpose(align, 0, 1)) # shape (T, M, K)

	embed_diff = (w_samples.unsqueeze(1) - traj_embeds.unsqueeze(0)).reshape(M*T, dim) # shape (M*T, dim)
	align = embed_diff @ l_samples.reshape(M*K, dim).T # shape (M*T, M*K)
	probs = np.maximum(align.reshape(M, T, -1).transpose(0, 1), 1e-8) # shape (T, M, M*K), so: (trajectory, weight, language)
	# probs += torch.abs(torch.min(probs, dim=-1)[0].reshape(T, M, 1)) # normalize values to make everything positive
	probs /= torch.sum(probs, dim=-1).unsqueeze(-1) # shape (T, M, M*K)
	# tmp = torch.nn.functional.relu(M * probs / torch.sum(probs, dim=1).unsqueeze(1)) + eps
	tmp = M * probs / torch.sum(probs, dim=1).unsqueeze(1)
	f_values = torch.sum(torch.sum(torch.log2(tmp), dim=2), dim=1) / (M*K) # shape (T)
	
	idx = torch.argmax(f_values) # easy code uses argmin but also divides by negative M for some reason
	ig = torch.abs(torch.nan_to_num(f_values[idx], nan=1))
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