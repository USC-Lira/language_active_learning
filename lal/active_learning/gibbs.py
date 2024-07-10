import torch
import numpy as np
# import cupy as cp

def gibbs_w(queries, latent_dim, prev_w, seed=0):
    '''
    This is the Gibbs algorithm, performing to sample reward weights from the reward space. It only performs one step at a time to allow for parallelization.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        prev_w (type torch.tensor): the initial or most recently sampled reward weight
        seed (type int): the random seed
    returns:
        w (type torch.tensor): the next sampled reward weight w
    '''
    def logp(i, w):
        traj_embed = queries[i][0]
        feedback_embed = queries[i][1]
        # return torch.log(1 / (1 + torch.exp(-(w @ feedback_embed))))
        return (feedback_embed @ (w - traj_embed)).item() # new model propto BT model using cosine similarity

    def logprob(w):
        if w.norm(2) > 1: return -np.inf
        return np.sum([logp(i, w) for i in range(len(queries))])

    w = prev_w

    # k = 20
    # chosen = np.random.choice(np.arange(latent_dim), k, replace=False)

    for i in range(latent_dim):
    # for i in chosen:
        # Propose a new sample, hoping its within the bounds [-1, 1]
        w_new = w.clone()
        lower_bound = -0.25 # originally had it at 1 but change to this for now
        upper_bound = 0.25
        w_new[i] = torch.FloatTensor(1).uniform_(lower_bound, upper_bound)

        # Calculate acceptance probability
        current_log_prob = logprob(w)
        new_log_prob = logprob(w_new)
        acceptance_prob = np.exp(new_log_prob - current_log_prob)
        
        # Accept or reject the new sample
        if np.random.rand(1) < acceptance_prob:
            w[i] = w_new[i]    
    return w

def gibbs_l(queries, w, latent_dim, num_l_samples, burn_in=1000, thin=50, seed=0, k=20):
    '''
    This is the Gibbs algorithm, performing to sample language from embedding space.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        w (type torch.tensor): the sampled reward model weights that is needed to then sample language
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_l_samples (type int): the number of l's that are sampled
        burn_in (type int): the number of samples to burn
        thin (type int): the increment that we take from the samples to thin out the set
        seed (type int): the random seed
    returns:
        torch.stack(l_samples) (type torch.tensor): the set of l samples
    '''
    def logp(i, l):
        traj_embed = queries[i][0]
        return (l @ (w - traj_embed)).item() # new model propto BT model using cosine similarity
	
    def logprob(l):
        if l.norm(2) > 1: return -np.inf
        return np.sum([logp(i, l) for i in range(len(queries))])

    l_samples = []
    l = torch.zeros(latent_dim)

    for _ in range(num_l_samples):
        # k = k
        # chosen = np.sort(np.random.choice(np.arange(latent_dim), k, replace=False))

        for i in range(latent_dim):
        # for i in chosen:
            # Sample from the conditional distribution for l[i] within the bounds [-1, 1]
            current_log_prob = logprob(l)
            
            l_new = l.clone()
            lower_bound = -0.25 # originally had it at 1 but change to this for now
            upper_bound = 0.25
            l_new[i] = torch.FloatTensor(1).uniform_(lower_bound, upper_bound)
            
            new_log_prob = logprob(l_new)
            acceptance_prob = np.exp(new_log_prob - current_log_prob)
            
            # Accept or reject the new sample
            if np.random.rand(1) < acceptance_prob:
                l[i] = l_new[i]
        
        if _ >= burn_in and _ % thin == 0: l_samples.append(l.clone())

    return torch.stack(l_samples)
    
if __name__ == "__main__":
    import time
    latent_dim = 512
    prev_w = torch.randn(latent_dim)
    prev_w /= torch.norm(prev_w)
    prev_w /= torch.norm(prev_w) # do it twice to ensure norm is less than 1
    traj_embed = torch.randn(100, latent_dim)
    traj_embed /= torch.norm(traj_embed)
    feedback_embed = torch.randn(100, latent_dim)
    feedback_embed /= torch.norm(feedback_embed)
    queries = [[traj_embed[i], feedback_embed[i]] for i in range(10)]
    start_time = time.time()
    w = gibbs_w(queries, latent_dim, prev_w)
    print(time.time() - start_time) # 0.0038 sec
    # print(w)
    start_time = time.time()
    l = gibbs_l(queries, w, latent_dim, 150, 100, 5, k=20)
    print(time.time() - start_time) # 9 secs w/o random idx, 0.38 secs w random idxs and k=20