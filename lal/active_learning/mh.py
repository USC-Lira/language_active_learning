import numpy as np
from numba import njit

@njit
def mh(traj_embeds, feedback_embeds, latent_dim, prev_w, step_size=0.001, num_w_samples=500, burn_in_w=5000, thin_w=250, 
    num_l_samples=10, burn_in_l=100, thin_l=5):

    def mh_w_helper(traj_embeds, feedback_embeds, latent_dim, prev_w, step_size=0.001):
        '''
        This is the Metropolis-Hastings algorithm, performing to sample reward weights from the reward space. It only performs one step at a time to allow for parallelization.
        parameters:
            queries (type list): the list of all query feedbacks provided by the user
            latent_dim (type int): the dimension of the latent space of the embeddings
            prev_w (type torch.tensor): the initial or most recently sampled reward weight
            step_size (type float): the step size for the proposal fxn in Metropolis-Hastings
            burn_in (type int): the number of samples to burn
            thin (type int): the increment that we take from the samples to thin out the set
        returns:
            w (type torch.tensor): the next sampled reward weight w
        '''
        def logp(i, w):
            feedback_embed = feedback_embeds[i].reshape(1, 512)
            diff = (w - traj_embeds[i]).reshape(512, 1)
            # return (feedback_embed @ (w - traj_embed)).item() # new model propto BT model using cosine similarity
            return np.dot(feedback_embed, diff).item() # new model propto BT model using cosine similarity
        
        def logprob(w):
            if np.linalg.norm(w) > 1: return -np.inf
            log_prob = np.float32(0.)
            for i in range(traj_embeds.shape[0]-1): 
                log_prob += logp(i, w)
            return log_prob

        # Propose a new sample, hoping its within the bounds [-1, 1]
        w_new = (prev_w + np.random.uniform(-step_size, step_size, latent_dim)).astype(np.float32)
        # Calculate acceptance probability
        current_log_prob = logprob(prev_w)
        new_log_prob = logprob(w_new)
        acceptance_prob = np.exp(new_log_prob - current_log_prob)

        # Accept or reject the new sample
        if np.random.rand(1) < acceptance_prob:
            return w_new
        return prev_w

    def mh_l_helper(traj_embeds, w, latent_dim, num_l_samples, step_size=0.03, burn_in=1000, thin=50):
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

        for idx in range(num_l_samples):
            # Propose a new sample, hoping its within the bounds [-1, 1]
            l_new = (l + np.random.uniform(-step_size, step_size, latent_dim)).astype(np.float32)
        
            # Calculate acceptance probability
            current_log_prob = logprob(l)
            new_log_prob = logprob(l_new)
            acceptance_prob = np.exp(new_log_prob - current_log_prob)
            
            # Accept or reject the new sample
            if np.random.rand(1) < acceptance_prob: l = l_new.copy()
            if idx >= burn_in and idx % thin == 0: l_samples.append(l.copy())
        
        return l_samples

    w_samples = []
    l_samples = []    
    for i in range(num_w_samples * thin_w + burn_in_w):
        w = mh_w_helper(traj_embeds, feedback_embeds, latent_dim, prev_w) # do one step in mcmc to sample w
        prev_w = w.copy()
        if i >= burn_in_w and i % thin_w == 0:
            w_samples.append(prev_w)
            l_samples.append(mh_l_helper(traj_embeds, prev_w, latent_dim, num_l_samples * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l))
    return w_samples, l_samples

@njit
def mh_w(traj_embeds, feedback_embeds, latent_dim, prev_w, step_size=0.03):
# def mh_w(traj_embeds, feedback_embeds, latent_dim, prev_w, step_size=0.001):
    '''
    This is the Metropolis-Hastings algorithm, performing to sample reward weights from the reward space. It only performs one step at a time to allow for parallelization.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        prev_w (type torch.tensor): the initial or most recently sampled reward weight
        step_size (type float): the step size for the proposal fxn in Metropolis-Hastings
        burn_in (type int): the number of samples to burn
        thin (type int): the increment that we take from the samples to thin out the set
    returns:
        w (type torch.tensor): the next sampled reward weight w
    '''
    def logp(i, w):
        feedback_embed = feedback_embeds[i].reshape(1, 512)
        diff = (w - traj_embeds[i]).reshape(512, 1)
        # return (feedback_embed @ (w - traj_embed)).item() # new model propto BT model using cosine similarity
        return np.dot(feedback_embed, diff).item() # new model propto BT model using cosine similarity
	
    def logprob(w):
        if np.linalg.norm(w) > 1: return -np.inf
        log_prob = np.float32(0.)
        for i in range(traj_embeds.shape[0]-1): 
            log_prob += logp(i, w)
        return log_prob

    # Propose a new sample, hoping its within the bounds [-1, 1]
    w_new = (prev_w + np.random.uniform(-step_size, step_size, latent_dim)).astype(np.float32)
    # Calculate acceptance probability
    current_log_prob = logprob(prev_w)
    new_log_prob = logprob(w_new)
    acceptance_prob = np.exp(new_log_prob - current_log_prob)

    # Accept or reject the new sample
    if np.random.rand(1) < acceptance_prob:
        return w_new
    return prev_w

@njit
def mh_l(traj_embeds, w, latent_dim, num_l_samples, step_size=0.03, burn_in=1000, thin=50):
# def mh_l(traj_embeds, w, latent_dim, num_l_samples, step_size=0.001, burn_in=1000, thin=50):
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

    for idx in range(num_l_samples):
        # Propose a new sample, hoping its within the bounds [-1, 1]
        l_new = (l + np.random.uniform(-step_size, step_size, latent_dim)).astype(np.float32)
    
        # Calculate acceptance probability
        current_log_prob = logprob(l)
        new_log_prob = logprob(l_new)
        acceptance_prob = np.exp(new_log_prob - current_log_prob)
        
        # Accept or reject the new sample
        if np.random.rand(1) < acceptance_prob: l = l_new.copy()
        if idx >= burn_in and idx % thin == 0: l_samples.append(l.copy())
    
    return l_samples

if __name__ == "__main__":
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
    
    num_w_samples=100
    burn_in_w=5000
    thin_w=250
    num_l_samples=100
    burn_in_l=5000
    thin_l=250
    
    w_samples = []
    l_samples = []
    start_time = time.time()
    for i in range(num_w_samples * thin_w + burn_in_w):
        w = mh_w(traj_embeds, feedback_embeds, latent_dim, prev_w1) # do one step in mcmc to sample w
        prev_w1 = w.copy()
        if i >= burn_in_w and i % thin_w == 0:
            w_samples.append(prev_w1)
            l_samples.append(mh_l(traj_embeds, prev_w1, latent_dim, num_l_samples * thin_l + burn_in_l, burn_in=burn_in_l, thin=thin_l))
    print(f"What we currently do: {time.time() - start_time}") # 

    # start_time = time.time()
    # w_samples, l_samples = mh(traj_embeds, feedback_embeds, latent_dim, prev_w2, num_w_samples=num_w_samples, burn_in_w=burn_in_w,
    #     thin_w=thin_w, num_l_samples=num_l_samples, burn_in_l=burn_in_l, thin_l=thin_l)
    # print(f"What we might change to: {time.time() - start_time}") # 