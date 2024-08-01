import numpy as np
from numba import njit

# @njit
def gibbs_w(traj_embeds, feedback_embeds, latent_dim, prev_w, k=50, seed=0):
    '''
    This is the Gibbs algorithm, performing to sample reward weights from the reward space. It only performs one step at a time to allow for parallelization.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        prev_w (type torch.tensor): the initial or most recently sampled reward weight
    returns:
        w (type torch.tensor): the next sampled reward weight w
    '''
    # np.random.seed(seed)
    traj_embeds = traj_embeds[1:]
    feedback_embeds = feedback_embeds[1:]

    def logp(i, w):
        feedback_embed = feedback_embeds[i].reshape(1, 512)
        diff = (w - traj_embeds[i]).reshape(512, 1)
        # return (feedback_embed @ (w - traj_embed)).item() # new model propto BT model using cosine similarity
        return np.dot(feedback_embed, diff).item() # new model propto BT model using cosine similarity
	
    def logprob(w):
        print(np.linalg.norm(w))
        if np.linalg.norm(w) > 1: return -np.inf
        log_prob = np.float32(0.)
        for i in range(traj_embeds.shape[0]-1): 
            log_prob += logp(i, w)
        return log_prob

    w = prev_w.copy()

    chosen = np.random.choice(np.arange(latent_dim), k, replace=False)
    lower_bound = -0.2 # originally had it at 1 but change to this for now
    # lower_bound = -1
    upper_bound = 0.2
    # upper_bound = 1

    # for i in range(latent_dim):
    for i in chosen:
        # Propose a new sample, hoping its within the bounds [-1, 1]
        w_new = w.copy()
        w_new[i] = np.random.uniform(lower_bound, upper_bound, 1).reshape(-1, 1).astype(np.float32).item()

        # Calculate acceptance probability
        current_log_prob = logprob(w)
        new_log_prob = logprob(w_new)
        acceptance_prob = np.exp(new_log_prob - current_log_prob)
        
        # Accept or reject the new sample
        if np.random.rand(1) < acceptance_prob:
            w[i] = w_new[i]    
    return w

# @njit
def gibbs_l(traj_embeds, w, latent_dim, num_l_samples, burn_in=1000, thin=50, k=20):
    '''
    This is the Gibbs algorithm, performing to sample language from embedding space.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        w (type torch.tensor): the sampled reward model weights that is needed to then sample language
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_l_samples (type int): the number of l's that are sampled
        burn_in (type int): the number of samples to burn
        thin (type int): the increment that we take from the samples to thin out the set
    returns:
        torch.stack(l_samples) (type torch.tensor): the set of l samples
    '''
    def logp(i, l):
        diff = (w - traj_embeds[i]).reshape(512, 1).astype(np.float32)
        return np.dot(l.reshape(1, 512), diff).item() # new model propto BT model using cosine similarity

    def logprob(l):
        if np.linalg.norm(l) > 1: return -np.inf
        log_prob = np.float32(0.)
        for i in range(traj_embeds.shape[0]-1): 
            log_prob += logp(i, l)
        return log_prob

    l_samples = []
    l = np.zeros(latent_dim, dtype=np.float32)
    lower_bound = -0.25 # originally had it at 1 but change to this for now
    upper_bound = 0.25

    for _ in range(num_l_samples):
        chosen = np.sort(np.random.choice(np.arange(latent_dim), k, replace=False))

        # for i in range(latent_dim):
        for i in chosen:
            # Sample from the conditional distribution for l[i] within the bounds [-1, 1]
            l_new = l.copy()
            l_new[i] = np.random.uniform(lower_bound, upper_bound, 1).reshape(-1, 1).astype(np.float32).item()
            
            # Calculate acceptance probability
            current_log_prob = logprob(l)
            new_log_prob = logprob(l_new)
            acceptance_prob = np.exp(new_log_prob - current_log_prob)
            
            # Accept or reject the new sample
            if np.random.rand(1) < acceptance_prob:
                l[i] = l_new[i]
        
        if _ >= burn_in and _ % thin == 0: l_samples.append(l.copy())

    return l_samples
    
if __name__ == "__main__":
    np.random.seed(1)
    import time
    latent_dim = 512
    prev_w = np.random.randn(latent_dim).astype(np.float32)
    prev_w /= np.linalg.norm(prev_w)
    prev_w /= np.linalg.norm(prev_w) # do it twice to ensure norm is less than 1
    prev_w1 = prev_w.copy()
    prev_w2 = prev_w.copy()
    traj_embeds = np.random.randn(100, latent_dim)
    traj_embeds /= np.linalg.norm(traj_embeds)
    feedback_embeds = np.random.randn(100, latent_dim)
    feedback_embeds /= np.linalg.norm(feedback_embeds)
    # start_time = time.time()
    # w = mh_w(traj_embeds, feedback_embeds, latent_dim, prev_w) # one w takes 2 seconds w/ njit
    # w = mh_w(traj_embeds, feedback_embeds, latent_dim, prev_w) # one w takes 0.2 seconds w/o njit
    # print(time.time() - start_time) # 0.0038 sec
    # print(w)
    # start_time = time.time()
    # l = mh_l(traj_embeds, w, latent_dim, 100, 0.03, 1000, 50) # 100 l_samples takes 9 seconds w/ njit
    # l = mh_l(traj_embeds, w, latent_dim, 100, 1000, 50) # 100 l_samples takes 20 seconds w/o njit
    
    # so each w+100l takes ~12 seconds, then multiply by 100w's, so it'd take 1200 seconds or 20 mins w/ njit
    # 
    # print(time.time() - start_time) # 9 secs w/o random idx, 0.38 secs w random idxs and k=20
    
    num_w_samples=10
    burn_in_w=100
    thin_w=5
    num_l_samples=100
    burn_in_l=1000
    thin_l=50
    
    # w_samples = []
    # l_samples = []
    # start_time = time.time()
    # for i in range(num_w_samples * thin_w + burn_in_w):
    #     w = gibbs_w(traj_embeds, feedback_embeds, latent_dim, prev_w1) # do one step in mcmc to sample w
    #     prev_w1 = w.copy()
    #     if i >= burn_in_w and i % thin_w == 0:
    #         w_samples.append(prev_w1)
    #         l_samples.append(gibbs_l(traj_embeds, prev_w1, latent_dim, num_l_samples * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l))
    # print(f"What we currently do: {time.time() - start_time}") # 

    start_time = time.time()
    w_samples, l_samples = gibbs_njit(traj_embeds, feedback_embeds, latent_dim, prev_w2, num_w_samples=num_w_samples, burn_in_w=burn_in_w,
        thin_w=thin_w, num_l_samples=num_l_samples, burn_in_l=burn_in_l, thin_l=thin_l)
    print(f"Gibbs njit: {time.time() - start_time}") # 53 seconds

    start_time = time.time()
    w_samples, l_samples = gibbs_parallel(traj_embeds, feedback_embeds, latent_dim, prev_w2, num_w_samples=num_w_samples, burn_in_w=burn_in_w,
        thin_w=thin_w, num_l_samples=num_l_samples, burn_in_l=burn_in_l, thin_l=thin_l)
    print(f"Gibbs parallel: {time.time() - start_time}") # 57 seconds

    start_time = time.time()
    w_samples, l_samples = gibbs(traj_embeds, feedback_embeds, latent_dim, prev_w2, num_w_samples=num_w_samples, burn_in_w=burn_in_w,
        thin_w=thin_w, num_l_samples=num_l_samples, burn_in_l=burn_in_l, thin_l=thin_l)
    print(f"Gibbs: {time.time() - start_time}") # 51 seconds


# archived
# from concurrent.futures import ProcessPoolExecutor

# def gibbs_parallel(traj_embeds, feedback_embeds, latent_dim, prev_w, num_w_samples=100, burn_in_w=1000, thin_w=50, 
#     num_l_samples=10, burn_in_l=100, thin_l=5):
#     w_samples = []
#     l_samples = []
#     # with ThreadPoolExecutor() as executor: # parallelize the l sampling
#     with ProcessPoolExecutor(max_workers=num_cpus) as executor: # parallelize the l sampling
#         for i in range(num_w_samples * thin_w + burn_in_w):
#             w = gibbs_w(traj_embeds, feedback_embeds, latent_dim, prev_w) # do one step in mcmc to sample w
#             prev_w = w.copy()
#             if i >= burn_in_w and i % thin_w == 0:
#                 w_samples.append(prev_w)
#                 future = executor.submit(gibbs_l, traj_embeds, prev_w, latent_dim, num_l_samples * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l)
#                 l_samples.append(future.result())
#     return w_samples, l_samples

# def gibbs(traj_embeds, feedback_embeds, latent_dim, prev_w, num_w_samples=100, burn_in_w=1000, thin_w=50, 
#     num_l_samples=10, burn_in_l=100, thin_l=5):
#     # with ThreadPoolExecutor() as executor: # parallelize the l sampling
#     w_samples = []
#     l_samples = []
#     for i in range(num_w_samples * thin_w + burn_in_w):
#         w = gibbs_w(traj_embeds, feedback_embeds, latent_dim, prev_w) # do one step in mcmc to sample w
#         prev_w = w.copy()
#         if i >= burn_in_w and i % thin_w == 0:
#             w_samples.append(prev_w)
#             l_samples.append(gibbs_l(traj_embeds, prev_w, latent_dim, num_l_samples * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l))
#     return w_samples, l_samples

# @njit()
# def gibbs_njit(traj_embeds, feedback_embeds, latent_dim, prev_w, num_w_samples=100, burn_in_w=1000, thin_w=50, 
#     num_l_samples=10, burn_in_l=100, thin_l=5):

#     def gibbs_w_helper(traj_embeds, feedback_embeds, latent_dim, prev_w, k=20):
#         '''
#         This is the Gibbs algorithm, performing to sample reward weights from the reward space. It only performs one step at a time to allow for parallelization.
#         parameters:
#             queries (type list): the list of all query feedbacks provided by the user
#             latent_dim (type int): the dimension of the latent space of the embeddings
#             prev_w (type torch.tensor): the initial or most recently sampled reward weight
#         returns:
#             w (type torch.tensor): the next sampled reward weight w
#         '''
#         def logp(i, w):
#             feedback_embed = feedback_embeds[i].reshape(1, 512)
#             diff = (w - traj_embeds[i]).reshape(512, 1)
#             # return (feedback_embed @ (w - traj_embed)).item() # new model propto BT model using cosine similarity
#             return np.dot(feedback_embed, diff).item() # new model propto BT model using cosine similarity
        
#         def logprob(w):
#             if np.linalg.norm(w) > 1: return -np.inf
#             log_prob = np.float32(0.)
#             for i in range(traj_embeds.shape[0]-1): 
#                 log_prob += logp(i, w)
#             return log_prob

#         w = prev_w.copy()

#         chosen = np.random.choice(np.arange(latent_dim), k, replace=False)
#         lower_bound = -0.25 # originally had it at 1 but change to this for now
#         upper_bound = 0.25

#         # for i in range(latent_dim):
#         for i in chosen:
#             # Propose a new sample, hoping its within the bounds [-1, 1]
#             w_new = w.copy()
#             w_new[i] = np.random.uniform(lower_bound, upper_bound, 1).reshape(-1, 1).astype(np.float32).item()

#             # Calculate acceptance probability
#             current_log_prob = logprob(w)
#             new_log_prob = logprob(w_new)
#             acceptance_prob = np.exp(new_log_prob - current_log_prob)
            
#             # Accept or reject the new sample
#             if np.random.rand(1) < acceptance_prob:
#                 w[i] = w_new[i]    
#         return w

#     def gibbs_l_helper(traj_embeds, w, latent_dim, num_l_samples, burn_in=1000, thin=50, k=20):
#         '''
#         This is the Gibbs algorithm, performing to sample language from embedding space.
#         parameters:
#             queries (type list): the list of all query feedbacks provided by the user
#             w (type torch.tensor): the sampled reward model weights that is needed to then sample language
#             latent_dim (type int): the dimension of the latent space of the embeddings
#             num_l_samples (type int): the number of l's that are sampled
#             burn_in (type int): the number of samples to burn
#             thin (type int): the increment that we take from the samples to thin out the set
#         returns:
#             torch.stack(l_samples) (type torch.tensor): the set of l samples
#         '''
#         def logp(i, l):
#             diff = (w - traj_embeds[i]).reshape(512, 1).astype(np.float32)
#             return np.dot(l.reshape(1, 512), diff).item() # new model propto BT model using cosine similarity

#         def logprob(l):
#             if np.linalg.norm(l) > 1: return -np.inf
#             log_prob = np.float32(0.)
#             for i in range(traj_embeds.shape[0]-1): 
#                 log_prob += logp(i, l)
#             return log_prob

#         l_samples = []
#         l = np.zeros(latent_dim, dtype=np.float32)
#         lower_bound = -0.25 # originally had it at 1 but change to this for now
#         upper_bound = 0.25

#         for _ in range(num_l_samples):
#             chosen = np.sort(np.random.choice(np.arange(latent_dim), k, replace=False))

#             # for i in range(latent_dim):
#             for i in chosen:
#                 # Sample from the conditional distribution for l[i] within the bounds [-1, 1]
#                 l_new = l.copy()
#                 l_new[i] = np.random.uniform(lower_bound, upper_bound, 1).reshape(-1, 1).astype(np.float32).item()
                
#                 # Calculate acceptance probability
#                 current_log_prob = logprob(l)
#                 new_log_prob = logprob(l_new)
#                 acceptance_prob = np.exp(new_log_prob - current_log_prob)
                
#                 # Accept or reject the new sample
#                 if np.random.rand(1) < acceptance_prob:
#                     l[i] = l_new[i]
            
#             if _ >= burn_in and _ % thin == 0: l_samples.append(l.copy())

#         return l_samples
    
#     w_samples = []
#     l_samples = []    
#     for i in range(num_w_samples * thin_w + burn_in_w):
#         w = gibbs_w_helper(traj_embeds, feedback_embeds, latent_dim, prev_w) # do one step in mcmc to sample w
#         prev_w = w.copy()
#         if i >= burn_in_w and i % thin_w == 0:
#             w_samples.append(prev_w)
#             l_samples.append(gibbs_l_helper(traj_embeds, prev_w, latent_dim, num_l_samples * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l))
#     return w_samples, l_samples
