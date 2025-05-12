import numpy as np
import scipy

def laplace_w(traj_embeds, feedback_embeds, latent_dim, num_w_samples, initial_w):
    '''
    This is the Laplace Approximation method, approximating the reward weight space as a gaussian, and then direct sampling from the approximation.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_w_samples (type int): the number of l's that are sampled
    returns:
        w_samples (type torch.tensor): the set of w samples
    '''
    traj_embeds = traj_embeds[1:]
    feedback_embeds = feedback_embeds[1:]

    def logp(i, w):
        feedback_embed = feedback_embeds[i].reshape(1, -1)
        diff = (w - traj_embeds[i]).reshape(-1, 1)
        return np.dot(feedback_embed, diff).item()
	
    def logprob(w):
        if np.linalg.norm(w) > 1: return -100
        log_prob = np.float32(0.)
        for i in range(traj_embeds.shape[0]): 
            log_prob += logp(i, w)
        return log_prob
        
    def neg_logprob(w):
        return -logprob(w)
    
    def prob_jacobian(w):
        return -np.sum(feedback_embeds, axis=0)

    def constraint_function(w):
        return 1 - np.linalg.norm(w)
    
    def constraint_jacobian(w):
        norm = np.linalg.norm(w)
        if norm == 0:
            return np.zeros_like(w)
        return -w / norm

    constraints = {'type': 'ineq', 'fun': constraint_function, 'jac': constraint_jacobian}
    solution = scipy.optimize.minimize(neg_logprob, initial_w, method='SLSQP', jac=prob_jacobian, constraints=constraints) # optimize using SLSQP to get the true mode
    mode = solution.x
    solution = scipy.optimize.minimize(neg_logprob, mode, method='L-BFGS-B', options={"maxiter":1}) # optimize using L-BFGS-B to get the approx hessian
    inv_hess = solution.hess_inv.todense() * 5e-2
    w_samples = np.random.multivariate_normal(np.nan_to_num(mode), np.nan_to_num(inv_hess), size=num_w_samples)
    return w_samples

def laplace_l(traj_embeds, w, latent_dim, num_l_samples):
    '''
    This is the Laplace Approximation method, approximating the embedding space as a gaussian, and then direct sampling from the approximation.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        w (type torch.tensor): the sampled reward model weights that is needed to then sample language
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_l_samples (type int): the number of l's that are sampled
    returns:
        l_samples (type torch.tensor): the set of l samples
    '''
    traj_embeds = traj_embeds[1:]

    def logp(i, l):
        diff = (w - traj_embeds[i]).reshape(-1, 1).astype(np.float32)
        # diff = (w - traj_embeds[i]/2).reshape(512, 1).astype(np.float32)
        return np.dot(l.reshape(1, -1), diff).item() # new model propto BT model using cosine similarity

    def logprob(l):
        # if np.linalg.norm(l) > 1: return -100
        log_prob = np.float32(0.)
        for i in range(traj_embeds.shape[0]): 
            log_prob += logp(i, l)
        return log_prob

    def neg_logprob(l):
        return -logprob(l)

    def constraint_function(l):
        return 1 - np.linalg.norm(l)

    constraints = {'type': 'ineq', 'fun': constraint_function}
    solution = scipy.optimize.minimize(neg_logprob, np.zeros(latent_dim), method='SLSQP', constraints=constraints) # optimize using SLSQP to get the true mode
    solution = scipy.optimize.minimize(neg_logprob, solution.x, method='L-BFGS-B', options={"maxiter":1}) # optimize using L-BFGS-B to get the approx hessian
    inv_hess = solution.hess_inv.todense() * 1e-4
    l_samples = np.random.multivariate_normal(np.nan_to_num(solution.x), np.nan_to_num(inv_hess), size=num_l_samples)
    print("l: ", np.linalg.norm(solution.x), np.linalg.norm(inv_hess))
    return l_samples

def laplace_l_sampling(traj_embeds, w_samples, latent_dim, num_w_samples, num_l_samples):
    '''
    This is the Laplace Approximation method, approximating the embedding space as a gaussian, and then direct sampling from the approximation.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        w (type torch.tensor): the sampled reward model weights that is needed to then sample language
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_l_samples (type int): the number of l's that are sampled
    returns:
        l_samples (type torch.tensor): the set of l samples
    '''
    traj_embeds = traj_embeds[1:]
    embed_diff = (w_samples.reshape(w_samples.shape[0], 1, w_samples.shape[1]) - traj_embeds.reshape(1, traj_embeds.shape[0], traj_embeds.shape[1]))

    def logprob(l):
        if np.linalg.norm(l) > 1: return -100
        align = embed_diff @ l
        return np.sum(align)

    def neg_logprob(l):
        return -logprob(l)

    def prob_jacobian(l):
        return -np.sum(embed_diff, axis=(0, 1))

    def constraint_function(l):
        return 1 - np.linalg.norm(l)
    
    def constraint_jacobian(l):
        norm = np.linalg.norm(l)
        if norm == 0:
            return np.zeros_like(l)
        return -l / norm

    if len(traj_embeds) == 0: # uninformed prior
        guess = np.random.rand(latent_dim)
    else: # informed prior
        guess = embed_diff.mean(axis=1).mean(axis=0)
    guess = (guess / np.linalg.norm(guess)) * 0.95
    constraints = {'type': 'ineq', 'fun': constraint_function, 'jac': constraint_jacobian}
    solution = scipy.optimize.minimize(neg_logprob, guess, method='SLSQP', jac=prob_jacobian, constraints=constraints) # optimize using SLSQP to get the true mode
    mode = solution.x
    solution = scipy.optimize.minimize(neg_logprob, mode, method='L-BFGS-B', options={"maxiter":1}) # optimize using L-BFGS-B to get the approx hessian
    inv_hess = solution.hess_inv.todense() * 5e-2
    l_samples = np.random.multivariate_normal(np.nan_to_num(mode), np.nan_to_num(inv_hess), size=num_w_samples * num_l_samples)
    return l_samples.reshape(num_w_samples, num_l_samples, -1)

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
    num_w_samples=100
    num_l_samples=100
    start_time = time.time()
    w_samples = laplace_w(traj_embeds, feedback_embeds, latent_dim, num_w_samples)
    print(time.time() - start_time) # 2 seconds using 10 iterations for minimizing; 21.9 seconds for 21 iterations
    start_time = time.time()
    l_samples = laplace_l(traj_embeds, w_samples[50], latent_dim, num_l_samples)
    print(time.time() - start_time) # 12 seconds using 16 iterations