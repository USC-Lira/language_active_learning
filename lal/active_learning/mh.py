import torch
import numpy as np

def mh_w(queries, latent_dim, prev_w, step_size=0.001, seed=0):
    '''
    This is the Metropolis-Hastings algorithm, performing to sample reward weights from the reward space. It only performs one step at a time to allow for parallelization.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        prev_w (type torch.tensor): the initial or most recently sampled reward weight
        step_size (type float): the step size for the proposal fxn in Metropolis-Hastings
        burn_in (type int): the number of samples to burn
        thin (type int): the increment that we take from the samples to thin out the set
        seed (type int): the random seed
    returns:
        w (type torch.tensor): the next sampled reward weight w
    '''
    def logp(i, w):
        traj_embed = queries[i][0]
        feedback_embed = queries[i][1]
        # return torch.log(1 / (1 + torch.exp(-(w @ feedback_embed)))) # prev BT model
        return (feedback_embed @ (w - traj_embed)).item() # new model propto BT model using cosine similarity
	
    def logprob(w):
        if w.norm(2) > 1: return -np.inf
        return np.sum([logp(i, w) for i in range(len(queries))])

    w = prev_w

    # Propose a new sample, hoping its within the bounds [-1, 1]
    w_new = w + torch.FloatTensor(latent_dim).uniform_(-step_size, step_size)

    # Calculate acceptance probability
    current_log_prob = logprob(w)
    new_log_prob = logprob(w_new)
    acceptance_prob = np.exp(new_log_prob - current_log_prob)
    
    # Accept or reject the new sample
    if np.random.rand(1) < acceptance_prob:
        w = w_new
    
    return w

def mh_l(queries, w, latent_dim, num_l_samples, step_size=0.03, burn_in=1000, thin=50, seed=0):
    '''
    This is the Metropolis-Hastings algorithm, performing to sample language from embedding space.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        w (type torch.tensor): the sampled reward model weights that is needed to then sample language
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_l_samples (type int): the number of l's that are sampled
        step_size (type float): the step size for the proposal fxn in Metropolis-Hastings
        burn_in (type int): the number of samples to burn
        thin (type int): the increment that we take from the samples to thin out the set
        seed (type int): the random seed
    returns:
        torch.stack(l_samples) (type torch.tensor): the set of l samples
    '''
    # def logprob(l):
	# 	if l.norm(2) > 1:
	# 		return -np.inf
    #     return (w @ l) # this should be w @ (tau_embed + l)

    def logp(i, l):
        traj_embed = queries[i][0]
        return (l @ (w - traj_embed)).item() # new model propto BT model using cosine similarity
	
    def logprob(l):
        if l.norm(2) > 1: return -np.inf
        return np.sum([logp(i, w) for i in range(len(queries))])

    l_samples = []
    l = torch.zeros(latent_dim)

    for _ in range(num_l_samples):
        # Propose a new sample, hoping its within the bounds [-1, 1]
        l_new = l + torch.FloatTensor(latent_dim).uniform_(-step_size, step_size)
    
        # Calculate acceptance probability
        current_log_prob = logprob(l)
        new_log_prob = logprob(l_new)
        acceptance_prob = np.exp(new_log_prob - current_log_prob)
        
        # Accept or reject the new sample
        if np.random.rand(1) < acceptance_prob: l = l_new
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
    w = mh_w(queries, latent_dim, prev_w)
    print(time.time() - start_time) # 0.00172 sec
    # print(w)
    start_time = time.time()
    mh_l(queries, w, latent_dim, 150, step_size=0.03, burn_in=100, thin=5)
    print(time.time() - start_time) # 0.0162