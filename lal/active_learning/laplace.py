import torch
import numpy as np
import scipy

def laplace_w(queries, latent_dim, num_w_samples, seed=0):
    '''
    This is the Laplace Approximation method, approximating the reward weight space as a gaussian, and then direct sampling from the approximation.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_w_samples (type int): the number of l's that are sampled
        seed (type int): the random seed
    returns:
        w_samples (type torch.tensor): the set of w samples
    '''
    def logp(i, w):
        traj_embed = queries[i][0]
        feedback_embed = queries[i][1]
        # return torch.log(1 / (1 + torch.exp(-(w @ feedback_embed))))
        return (feedback_embed @ (torch.tensor(w) - traj_embed).to(torch.float)).item() # new model propto BT model using cosine similarity

    def logprob(w):
        # if w.norm(2) > 1: return -np.inf
        if np.linalg.norm(w) > 1: return -100
        return np.sum([logp(i, w) for i in range(len(queries))])

    def neg_logprob(w):
        return -logprob(w)

    solution = scipy.optimize.minimize(neg_logprob, np.zeros(latent_dim), options={"maxiter":10}) # optimize using initial guess
    mode = torch.tensor(solution.x, dtype=torch.float)
    mode /= torch.norm(mode) # normalize the mode
    inv_hess = torch.tensor(solution.hess_inv)
    normalized_inv_hess = inv_hess*1e-10
    lambs = torch.norm(normalized_inv_hess) # L2 regularization
    identity = torch.eye(latent_dim)
    normalized_inv_hess += lambs * identity
    covariance = normalized_inv_hess.to(torch.float)
    w_samples = torch.distributions.MultivariateNormal(torch.nan_to_num(mode), torch.nan_to_num(covariance)).sample((num_w_samples,))
    return w_samples

def laplace_l(queries, w, latent_dim, num_l_samples, seed=0):
    '''
    This is the Laplace Approximation method, approximating the embedding space as a gaussian, and then direct sampling from the approximation.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        w (type torch.tensor): the sampled reward model weights that is needed to then sample language
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_l_samples (type int): the number of l's that are sampled
        seed (type int): the random seed
    returns:
        l_samples (type torch.tensor): the set of l samples
    '''
    def logp(i, l):
        traj_embed = queries[i][0]
        return (torch.tensor(l).to(torch.float) @ (w - traj_embed)).item() # new model propto BT model using cosine similarity
	
    def logprob(l):
        # if l.norm(2) > 1: return -np.inf
        if np.linalg.norm(l) > 1: return -100
        return np.sum([logp(i, l) for i in range(len(queries))])

    def neg_logprob(l):
        return -logprob(l)

    solution = scipy.optimize.minimize(neg_logprob, np.zeros(latent_dim), options={"maxiter":10}) # optimize using initial guess
    mode = torch.tensor(solution.x, dtype=torch.float)
    mode /= torch.norm(mode) # normalize the mode
    inv_hess = torch.tensor(solution.hess_inv)
    normalized_inv_hess = inv_hess*1e-10
    lambs = torch.norm(normalized_inv_hess) # L2 regularization
    identity = torch.eye(latent_dim)
    normalized_inv_hess += lambs * identity
    covariance = normalized_inv_hess.to(torch.float)
    l_samples = torch.distributions.MultivariateNormal(torch.nan_to_num(mode), torch.nan_to_num(covariance)).sample((num_l_samples,))
    return l_samples

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
    w = laplace_w(queries, latent_dim, 100)[50]
    # print(time.time() - start_time) # 2 seconds using 10 iterations for minimizing; 21.9 seconds for 21 iterations
    start_time = time.time()
    l = laplace_l(queries, w, latent_dim, 10)
    # print(time.time() - start_time) # 12 seconds using 16 iterations