import torch
import numpy as np
from scipy.integrate import quad
from scipy.stats import multivariate_normal

def ep_w(queries, latent_dim, num_w_samples, seed=0):
    '''
    This is the Expectation Propagation method, approximating the reward weight space as a gaussian, and then direct sampling from the approximation.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_w_samples (type int): the number of l's that are sampled
        seed (type int): the random seed
    returns:
        w_samples (type torch.tensor): the set of w samples
    '''
    def compute_moments(mu, cov, psi, tau):
        sd = np.sqrt(cov)
        def f(w): # the tilted/hybrid distribution
            # if np.linalg.norm(w) > 1: return multivariate_normal.pdf(w, mu, sd) * -100
            return multivariate_normal.pdf(w, mu, sd) * np.exp(psi @ (w-tau)) / 100 # <- issue here
        z, _ = quad(f, -np.inf, np.inf)
        mu_new, _ = quad(lambda x: f(x) * x, -np.inf, np.inf) # definition of mu
        if z != 0: mu_new /= z # normalize mu
        cov_new, _ = quad(lambda x: f(x) * (x - mu_new) ** 2, -np.inf, np.inf) # definition of covariance
        return mu_new, cov_new
	
    if len(queries) < 2:
        mu = np.zeros(512)
        cov = np.eye(512)
        cov /= np.linalg.norm(cov)
        w_samples = np.random.multivariate_normal(mu, cov, num_w_samples)
        return torch.tensor(w_samples).to(torch.float)

    num_iterations = 3 # change to 100 later
    A = np.stack([np.eye(512) for q in range(len(queries))]) # total precision
    r = np.stack([np.zeros(512) for q in range(len(queries))]) # total shift
    # damping = 0.1

    for i in range(num_iterations):
        for q in range(len(queries)):
            # https://www.youtube.com/watch?v=0tomU1q3AdY resource
            # natural parameters of cavity distribution
            A_cav = np.sum(A, axis=0) - A[q] # cavity precision
            r_cav = np.sum(r, axis=0) - r[q] # cavity shift
            cov_cav = np.linalg.inv(A_cav) # cavity covariance, which is just the inverse of cavity precision
            mu_cav = cov_cav @ r_cav # cavity mu, which is just the covariance * cavity shift
            # import ipdb; ipdb.set_trace()

            # get tilted/hybrid
            mu_tilted, cov_tilted = compute_moments(mu_cav, cov_cav, queries[q][1], queries[q][0])

            # change back to natural params and update approx
            A[q] = np.linalg.inv(cov_tilted) - A_cav # new precision; feels weird to subtract cav but u do apparently
            r[q] = A[q] * mu_tilted - r_cav # new shift

    sigma = np.linalg.inv(np.sum(A, axis=0))
    mu = sigma @ np.sum(r, axis=0)
    w_samples = np.random.multivariate_normal(mu, sigma, num_w_samples)
    return torch.tensor(w_samples).to(torch.float)

    # def logprob(i, w):
    #     if w.norm(2) > 1: return -100
    #     traj_embed = queries[i][0]
    #     feedback_embed = queries[i][1]
    #     return (feedback_embed @ (w - traj_embed)).item() # new model propto BT model using cosine similarity

    # # Initial guess for EP approximation using a Gaussian
    # mode = torch.zeros(latent_dim)
    # precision = torch.eye(latent_dim) * 1e-4  # Initial precision (small value for stabilization)

    # # Number of EP iterations
    # num_iterations = 100

    # # Perform EP iterations
    # for iter in range(num_iterations):
    #     # Update Gaussian factors
    #     z = mode - torch.sqrt(torch.diag(precision)) # z-value
    #     m_cav = 1.0 / (1.0 + torch.exp(-z)) # sigmoid??
    #     s2 = 1.0 / (1.0 + 1.0 / torch.diag(precision)) # variance

    #     # Update factors
    #     mode_new = mode + torch.sqrt(s2) * (logprob(m_cav) / np.exp(logprob(m_cav)) - m_cav) # turns into -np.inf
    #     precision_new = torch.diag(1.0 / (s2 - s2 * logprob(m_cav) / np.exp(logprob(m_cav)) * (1.0 - logprob(m_cav)))) # turns into 0.

    #     # Damping to improve stability
    #     damping = 0.1
    #     mode = damping * mode_new + (1.0 - damping) * mode
    #     precision = damping * precision_new + (1.0 - damping) * precision

    # # mean = torch.nan_to_num(mode / torch.norm(mode))
    # mean = torch.nan_to_num(mode, posinf=0, neginf=0)
    # covariance = torch.inverse(precision) * 1e-6
    # # print(torch.norm(covariance))
    # w_samples = torch.distributions.MultivariateNormal(mean, covariance).sample((num_w_samples,))
    # # w_samples /= torch.norm(w_samples, dim=1).view(-1, 1)
    # return w_samples

def ep_l(queries, w, latent_dim, num_l_samples, burn_in=None, thin=None, seed=0):
    '''
    This is the Expectation Propagation method, approximating the embedding space as a gaussian, and then direct sampling from the approximation.
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
        return (l @ (w - traj_embed)).item() # new model propto BT model using cosine similarity
	
    def logprob(l):
        if l.norm(2) > 1: return -100
        return np.sum([logp(i, l) for i in range(len(queries))])

    # Set random seed for reproducibility
    # torch.manual_seed(seed)
    # np.random.seed(seed)

    # Initial guess for EP approximation
    mode = torch.zeros(latent_dim)
    precision = torch.eye(latent_dim) * 1e-4  # Initial precision (small value for stabilization)

    # Number of EP iterations
    num_iterations = 10

    # Perform EP iterations
    for iter in range(num_iterations):
        # Update Gaussian factors
        z = mode - torch.sqrt(torch.diag(precision))
        m = 1.0 / (1.0 + torch.exp(-z))
        s2 = 1.0 / (1.0 + 1.0 / torch.diag(precision))

        # Update cavity distribution parameters
        m_cav = m.clone()
        s2_cav = s2.clone()

        # Update factors
        mode_new = mode + torch.sqrt(s2) * (logprob(m_cav) / np.exp(logprob(m_cav)) - m_cav)
        precision_new = torch.diag(1.0 / (s2 - s2 * logprob(m_cav) / np.exp(logprob(m_cav)) * (1.0 - logprob(m_cav))))

        # Damping to improve stability
        damping = 0.1
        mode = damping * mode_new + (1.0 - damping) * mode
        precision = damping * precision_new + (1.0 - damping) * precision

    mean = torch.nan_to_num(mode / torch.norm(mode))
    l_samples = torch.distributions.MultivariateNormal(mean, torch.inverse(precision)).sample((num_l_samples,))
    return l_samples

if __name__ == "__main__":
    # import time
    latent_dim = 512
    prev_w = torch.randn(latent_dim)
    prev_w /= torch.norm(prev_w)
    prev_w /= torch.norm(prev_w) # do it twice to ensure norm is less than 1
    traj_embed = torch.randn(100, latent_dim)
    traj_embed /= torch.norm(traj_embed)
    feedback_embed = torch.randn(100, latent_dim)
    feedback_embed /= torch.norm(feedback_embed)
    queries = [[traj_embed[i], feedback_embed[i]] for i in range(10)]
    # start_time = time.time()
    w = ep_w(queries, latent_dim, 100)[50]
    # print(time.time() - start_time) # 1.29 seconds for 10 iterations, 3.3 for 50 iterations, 5.89 for 100 iterations
    # start_time = time.time()
    l = ep_l(queries, w, latent_dim, 10)
    # print(time.time() - start_time) # 1.28 seconds