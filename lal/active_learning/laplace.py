import numpy as np
import scipy

def laplace_w(traj_embeds, feedback_embeds, latent_dim, num_w_samples):
    '''
    This is the Laplace Approximation method, approximating the reward weight space as a gaussian, and then direct sampling from the approximation.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_w_samples (type int): the number of l's that are sampled
    returns:
        w_samples (type torch.tensor): the set of w samples
    '''
    def logp(i, w):
        feedback_embed = feedback_embeds[i].reshape(1, 512)
        diff = (w - traj_embeds[i]).reshape(512, 1)
        return np.dot(feedback_embed, diff).item() # new model propto BT model using cosine similarity
        # diff = w - traj_embeds[i]
        # total_prob = np.exp(np.dot(feedback_embeds, diff))
        # return np.log(total_prob[i]/np.sum(total_prob))

    def logq(i, w):
        return np.log(1/(1 + np.exp(-(w @ feedback_embeds[i]))))
        # total_prob = np.exp(w.reshape(512 ,) @ (np.tile(traj_embeds[i], [traj_embeds.shape[0], 1]) + feedback_embeds).reshape(-1, 512).T) # size 20
        # return np.log(total_prob[i]/np.sum(total_prob))
	
    def logprob(w):
        if np.linalg.norm(w) > 1: return -100
        log_prob = np.float32(0.)
        for i in range(traj_embeds.shape[0]): 
            log_prob += logp(i, w)
            # log_prob += logq(i, w)
        return log_prob

    def neg_logprob(w):
        return -logprob(w)

    def constraint_function(w):
        return 1 - np.linalg.norm(w)

    traj_embeds = traj_embeds[1:]
    feedback_embeds = feedback_embeds[1:]

    constraints = {'type': 'ineq', 'fun': constraint_function}
    solution = scipy.optimize.minimize(neg_logprob, np.zeros(latent_dim), method='SLSQP', constraints=constraints) # optimize using SLSQP to get the true mode
    solution = scipy.optimize.minimize(neg_logprob, solution.x, method='L-BFGS-B', options={"maxiter":1}) # optimize using L-BFGS-B to get the approx hessian
    # solution = scipy.optimize.minimize(neg_logprob, solution.x, method='L-BFGS-B', options={"maxiter":10}) # optimize using L-BFGS-B to get the approx hessian
    # https://github.com/rickecon/StructEst_W17/issues/26
    mode = solution.x / np.maximum(np.linalg.norm(solution.x), 1e-8)
    # mode = solution.x
    inv_hess = solution.hess_inv.todense() * 1e-2
    # inv_hess = solution.hess_inv.todense() * 1e-10
    # lambs = np.linalg.norm(inv_hess) # L2 regularization
    # inv_hess += lambs * np.eye(latent_dim)
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
    def logp(i, l):
        diff = (w - traj_embeds[i]).reshape(512, 1).astype(np.float32)
        return np.dot(l.reshape(1, 512), diff).item() # new model propto BT model using cosine similarity

    def logprob(l):
        if np.linalg.norm(l) > 1: return -100
        log_prob = np.float32(0.)
        for i in range(traj_embeds.shape[0]-1): 
            log_prob += logp(i, l)
        return log_prob

    def neg_logprob(l):
        return -logprob(l)

    def constraint_function(l):
        return 1 - np.linalg.norm(l)

    constraints = {'type': 'ineq', 'fun': constraint_function}
    solution = scipy.optimize.minimize(neg_logprob, np.zeros(latent_dim), method='SLSQP', constraints=constraints) # optimize using SLSQP to get the true mode
    solution = scipy.optimize.minimize(neg_logprob, solution.x, method='L-BFGS-B', options={"maxiter":10}) # optimize using L-BFGS-B to get the approx hessian
    mode = solution.x / np.linalg.norm(solution.x)
    inv_hess = solution.hess_inv.todense() * 1e-10
    lambs = np.linalg.norm(inv_hess) # L2 regularization
    inv_hess += lambs * np.eye(latent_dim)
    l_samples = np.random.multivariate_normal(np.nan_to_num(mode), np.nan_to_num(inv_hess), size=num_l_samples)
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
    num_w_samples=100
    num_l_samples=100
    start_time = time.time()
    w_samples = laplace_w(traj_embeds, feedback_embeds, latent_dim, num_w_samples)
    print(time.time() - start_time) # 2 seconds using 10 iterations for minimizing; 21.9 seconds for 21 iterations
    start_time = time.time()
    l_samples = laplace_l(traj_embeds, w_samples[50], latent_dim, num_l_samples)
    print(time.time() - start_time) # 12 seconds using 16 iterations