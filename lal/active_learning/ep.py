import torch
import numpy as np
from numba import njit
import scipy

@njit
def multivariate_normal_pdf(x, mean, covariance):
    """
    Calculates the PDF of a multivariate normal distribution.

    Args:
        x: (N,) or (N, D) array_like. Points at which to evaluate the PDF.
           If 1-D, assumed to be for a single data point of D dimensions.
           If 2-D, each row is assumed to be a data point.
        mean: (D,) array_like. Mean of the distribution.
        covariance: (D, D) array_like. Covariance matrix of the distribution.

    Returns:
        (N,) array_like. PDF values at the given points.
    """
    # import ipdb; ipdb.set_trace()
    x = np.atleast_2d(x)
    n_dimensions = len(mean)

    # Calculate the determinant and inverse of the covariance matrix
    det_covariance = np.maximum(np.linalg.det(covariance), 1e-8)
    inv_covariance = np.linalg.pinv(covariance)

    # Calculate the exponent term of the PDF
    diff = x - mean
    exponent = -0.5 * np.sum(np.dot(diff, inv_covariance) * diff, axis=1)

    # Calculate the normalization constant
    normalization = 1.0 / ((2 * np.pi)**(n_dimensions / 2) * np.sqrt(det_covariance))

    return normalization * np.exp(np.minimum(exponent, 100))

def ep_w_dimension(traj_embeds, feedback_embeds, latent_dim, num_w_samples):
    '''
    This is the Expectation Propagation method, approximating the reward weight space as a gaussian, and then direct sampling from the approximation.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_w_samples (type int): the number of l's that are sampled
    returns:
        w_samples (type torch.tensor): the set of w samples
    '''
    if traj_embeds.shape[0] == 1: # this only occurs at iteration 0 when there has been no human query yet
        mu = np.zeros(latent_dim)
        cov = np.eye(latent_dim)
        cov /= np.linalg.norm(cov)
        w_samples = np.random.multivariate_normal(mu, cov, num_w_samples).reshape(num_w_samples, latent_dim)
        return w_samples

    def f(x, mu, cov, psi, tau):
        return scipy.stats.norm.pdf(x, mu, cov) * np.exp(x - psi @ tau)

    # following https://www.jmlr.org/papers/volume24/23-0104/23-0104.pdf page 7, algorithm 1
    # https://www.youtube.com/watch?v=0tomU1q3AdY resource

    num_iterations = 100
    Q_k = [np.eye(1) for q in range(traj_embeds.shape[0])]
    r_k = [np.zeros(1) for q in range(traj_embeds.shape[0])]
    Q_dot = np.sum([feedback_embeds[q].reshape(latent_dim, 1) @ Q_k[q] @ feedback_embeds[q].reshape(1, latent_dim) for q in range(feedback_embeds.shape[0])], axis=0) # (512, 512)
    r_dot = np.sum([feedback_embeds[q].reshape(latent_dim, 1) @ r_k[q] for q in range(feedback_embeds.shape[0])], axis=0) # (512,)
    eta = 0.01 # step size
    alpha = 0.3 # damping

    for i in range(num_iterations):
        Q_k_copy = Q_k.copy()
        r_k_copy = r_k.copy()
        Q_dot_copy = Q_dot.copy()
        r_dot_copy = r_dot.copy()
        Sigma_dot = np.linalg.pinv(Q_dot) # (512, 512)
        mu_dot = Sigma_dot @ r_dot # (512,)
        for q in range(traj_embeds.shape[0]):
            # EP Downdate
            Sigma_star = feedback_embeds[q].reshape(1, latent_dim) @ Sigma_dot @ feedback_embeds[q].reshape(latent_dim, 1) # (1,)
            mu_star = feedback_embeds[q].reshape(1, latent_dim) @ mu_dot # (1,)
            Q_star = np.linalg.pinv(Sigma_star) # (1,)
            r_star = Q_star @ mu_star # (1,)
            Q_cav = Q_star - eta * Q_k[q]
            r_cav = r_star - eta * r_k[q]
            Sigma_cav = np.abs(np.linalg.pinv(Q_cav))
            mu_cav = Sigma_cav @ r_cav

            # Tilted distribution inference
            zero_mom = np.maximum(scipy.integrate.quad(f, -1, 1, args=(mu_cav.item(), Sigma_cav.item(), feedback_embeds[q], traj_embeds[q]))[0], 1e-8)
            first_fxn = lambda x, mu, cov, psi, tau: f(x, mu, cov, psi, tau) * x
            first_mom = scipy.integrate.quad(first_fxn, -1, 1, args=(mu_cav.item(), Sigma_cav.item(), feedback_embeds[q], traj_embeds[q]))[0]
            second_fxn = lambda x, mu, cov, psi, tau: f(x, mu, cov, psi, tau) * x**2
            second_mom = scipy.integrate.quad(second_fxn, -1, 1, args=(mu_cav.item(), Sigma_cav.item(), feedback_embeds[q], traj_embeds[q]))[0]
            Sigma_tilted = np.array(second_mom / zero_mom - (first_mom / zero_mom) * (first_mom / zero_mom)).reshape(1, 1)
            mu_tilted = np.array(first_mom / zero_mom).reshape(1,)

            # EP update
            Q_tilted = np.linalg.pinv(Sigma_tilted)
            r_tilted = Q_tilted @ mu_tilted
            Q_tilde = (1 - alpha) * Q_k[q] + (alpha/eta) * (Q_tilted - Q_cav)
            r_tilde = (1 - alpha) * r_k[q] + (alpha/eta) * (r_tilted - r_cav)
            Q_sum = feedback_embeds[q].reshape(latent_dim, 1) @ (Q_tilde - Q_k[q]) @ feedback_embeds[q].reshape(1, latent_dim)
            Q_dot_copy += (Q_sum + Q_sum.T) / 2
            # Q_dot_copy += feedback_embeds[q].reshape(latent_dim, 1) @ (Q_tilde - Q_k[q]) @ feedback_embeds[q].reshape(1, latent_dim)
            r_dot_copy += feedback_embeds[q].reshape(latent_dim, 1) @ (r_tilde - r_k[q])
            Q_k_copy[q] = Q_tilde
            r_k_copy[q] = r_tilde

        Q_k = Q_k_copy.copy()
        r_k = r_k_copy.copy()
        Q_dot = Q_dot_copy.copy()
        r_dot = r_dot_copy.copy()

    final_sigma = np.linalg.pinv(Q_dot)
    # final_sigma /= np.linalg.norm(final_sigma)
    eig = np.real(np.linalg.eigvals(final_sigma))
    min_eig = np.min(eig)
    if min_eig < 0: 
        final_sigma -= 10*min_eig * np.eye(*final_sigma.shape)
    final_mu = final_sigma @ r_dot
    final_mu /= np.maximum(np.linalg.norm(final_mu), 1e-8)
    w_samples = np.random.multivariate_normal(final_mu, final_sigma, num_w_samples)
    return w_samples

def ep_w(traj_embeds, feedback_embeds, latent_dim, num_w_samples):
    '''
    This is the Expectation Propagation method, approximating the reward weight space as a gaussian, and then direct sampling from the approximation.
    parameters:
        queries (type list): the list of all query feedbacks provided by the user
        latent_dim (type int): the dimension of the latent space of the embeddings
        num_w_samples (type int): the number of l's that are sampled
    returns:
        w_samples (type torch.tensor): the set of w samples
    '''
    if traj_embeds.shape[0] == 1: # this only occurs at iteration 0 when there has been no human query yet
        mu = np.zeros(latent_dim)
        cov = np.eye(latent_dim)
        cov /= np.linalg.norm(cov)
        w_samples = np.random.multivariate_normal(mu, cov, num_w_samples).reshape(num_w_samples, latent_dim)
        return w_samples

    @njit
    def sample_hybrid(mu, cov, tau, psi):
        sd = np.sqrt(np.maximum(cov, 1e-8))
        def f(w): # the tilted/hybrid distribution
            return multivariate_normal_pdf(w, mu, sd) * np.exp((psi @ (w.astype(np.float32)-tau)).item())

        num_hybrid_samples = 100
        thin = 1
        burn_in = 0
        step_size = 0.001
        hybrid_samples = np.array([0.])
        h = np.zeros(latent_dim)
        for _ in range(num_hybrid_samples * thin + burn_in):
            # Propose a new sample, hoping its within the bounds [-1, 1]
            h_new = h + np.random.uniform(-step_size, step_size, latent_dim)

            # Calculate acceptance probability
            current_log_prob = np.maximum(f(h), 1e-8)
            new_log_prob = f(h_new)
            acceptance_prob = new_log_prob / current_log_prob
            
            # Accept or reject the new sample
            if np.random.rand(1) < acceptance_prob: h = h_new
            if _ >= burn_in and _ % thin == 0: hybrid_samples = np.append(hybrid_samples, h)
        return hybrid_samples

    num_iterations = 1
    A = np.stack([np.eye(latent_dim) for q in range(traj_embeds.shape[0])]) # total precision
    r = np.stack([np.zeros(latent_dim) for q in range(traj_embeds.shape[0])]) # total shift
    # damping = 0.1

    @njit
    def ep_iter():
        A_clone = np.copy(A)
        r_clone = np.copy(r)
        for i in range(num_iterations):
            # A_clone = np.copy(A)
            # r_clone = np.copy(r)
            for q in range(traj_embeds.shape[0]):
                # https://www.youtube.com/watch?v=0tomU1q3AdY resource
                # natural parameters of cavity distribution
                A_cav = np.sum(A_clone, axis=0) - A_clone[q] # cavity precision
                r_cav = np.sum(r_clone, axis=0) - r_clone[q] # cavity shift
                cov_cav = np.linalg.inv(A_cav) # cavity covariance, which is just the inverse of cavity precision
                mu_cav = cov_cav @ r_cav # cavity mu, which is just the covariance * cavity shift

                # get tilted/hybrid
                hybrid_samples = sample_hybrid(mu_cav, cov_cav, traj_embeds[q], feedback_embeds[q])[1:].reshape(-1, latent_dim)
                mu_tilted = np.sum(hybrid_samples, axis=0) / hybrid_samples.shape[0] #numba doesn't support .mean() w/ any args
                cov_tilted =  np.cov(hybrid_samples, rowvar=False)

                # change back to natural params and update approx
                # A_clone[q] = np.linalg.inv(cov_tilted) - A_cav # new precision; feels weird to subtract cav but u do apparently
                A_clone[q] = np.linalg.pinv(cov_tilted) - A_cav # new precision; feels weird to subtract cav but u do apparently
                # r_clone[q] = A_clone[q] @ mu_tilted - r_cav # new shift
                r_clone[q] = A_clone[q] @ mu_tilted - r_cav # new shift
            # A = np.copy(A_clone)
            # r = np.copy(r_clone)
        return A_clone, r_clone

    A, r = ep_iter()

    sigma = np.linalg.pinv(np.sum(A, axis=0))
    mu = sigma @ np.sum(r, axis=0)
    w_samples = np.random.multivariate_normal(mu, sigma, num_w_samples)
    return w_samples

def ep_l(traj_embeds, w, latent_dim, num_l_samples, burn_in=None, thin=None, seed=0):
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
    @njit
    def sample_hybrid(mu, cov, tau):
        sd = np.sqrt(np.maximum(cov, 1e-8))
        def f(l): # the tilted/hybrid distribution
            return multivariate_normal_pdf(l, mu, sd) * np.exp((l @ (w.astype(np.float32)-tau)).item())

        num_hybrid_samples = 100
        thin = 1
        burn_in = 0
        step_size = 0.001
        hybrid_samples = []
        h = np.zeros(latent_dim)
        for _ in range(num_hybrid_samples * thin + burn_in):
            # Propose a new sample, hoping its within the bounds [-1, 1]
            h_new = h + np.random.uniform(-step_size, step_size, latent_dim)

            # Calculate acceptance probability
            current_log_prob = np.maximum(f(h), 1e-8)
            new_log_prob = f(h_new)
            acceptance_prob = new_log_prob / current_log_prob
            
            # Accept or reject the new sample
            if np.random.rand(1) < acceptance_prob: h = h_new
            if _ >= burn_in and _ % thin == 0: hybrid_samples.append(h)
        return hybrid_samples

    if traj_embeds.shape[0] == 1: # this only occurs at iteration 0 when there has been no human query yet
        mu = np.zeros(latent_dim)
        cov = np.eye(latent_dim)
        cov /= np.linalg.norm(cov)
        w_samples = np.random.multivariate_normal(mu, cov, num_w_samples).reshape(num_w_samples, latent_dim)
        return w_samples

    num_iterations = 5
    A = np.stack([np.eye(latent_dim) for q in range(traj_embeds.shape[0])]) # total precision
    r = np.stack([np.zeros(latent_dim) for q in range(traj_embeds.shape[0])]) # total shift
    # damping = 0.1

    for i in range(num_iterations):
        # A_clone = np.copy(A)
        # r_clone = np.copy(r)
        for q in range(traj_embeds.shape[0]):
            # https://www.youtube.com/watch?v=0tomU1q3AdY resource
            # natural parameters of cavity distribution
            A_cav = np.sum(A, axis=0) - A[q] # cavity precision
            r_cav = np.sum(r, axis=0) - r[q] # cavity shift
            cov_cav = np.linalg.inv(A_cav) # cavity covariance, which is just the inverse of cavity precision
            mu_cav = cov_cav @ r_cav # cavity mu, which is just the covariance * cavity shift

            # get tilted/hybrid
            hybrid_samples = sample_hybrid(mu_cav, cov_cav, traj_embeds[q])
            mu_tilted = np.mean(hybrid_samples, axis=0)
            cov_tilted =  np.cov(hybrid_samples, rowvar=False)

            # change back to natural params and update approx
            # A_clone[q] = np.linalg.inv(cov_tilted) - A_cav # new precision; feels weird to subtract cav but u do apparently
            A[q] = np.linalg.pinv(cov_tilted) - A_cav # new precision; feels weird to subtract cav but u do apparently
            # r_clone[q] = A_clone[q] @ mu_tilted - r_cav # new shift
            r[q] = A[q] @ mu_tilted - r_cav # new shift
        # A = np.copy(A_clone)
        # r = np.copy(r_clone)

    sigma = np.linalg.pinv(np.sum(A, axis=0))
    mu = sigma @ np.sum(r, axis=0)
    l_samples = np.random.multivariate_normal(mu, sigma, num_l_samples)
    return l_samples

if __name__ == "__main__":
    import time
    latent_dim = 512
    prev_w = np.random.randn(latent_dim).astype(np.float32)
    prev_w /= np.linalg.norm(prev_w)
    prev_w /= np.linalg.norm(prev_w) # do it twice to ensure norm is less than 1
    prev_w1 = prev_w.copy()
    prev_w2 = prev_w.copy()
    traj_embeds = np.random.randn(20, latent_dim)
    traj_embeds /= np.linalg.norm(traj_embeds)
    feedback_embeds = np.random.randn(20, latent_dim)
    feedback_embeds /= np.linalg.norm(feedback_embeds)
    num_w_samples=100
    num_l_samples=100
    # start_time = time.time()
    # w_samples = ep_w(traj_embeds, feedback_embeds, latent_dim, num_w_samples)
    # print(time.time() - start_time) # 300 seconds using njit, 20 queries, 100 hybrid samples, 5 iterations
    start_time = time.time()
    w_samples = ep_w_dimension_new(traj_embeds, feedback_embeds, latent_dim, num_w_samples)
    print(f"{time.time() - start_time} seconds") # 300 seconds using njit, 20 queries, 100 hybrid samples, 5 iterations
    print(np.linalg.norm(w_samples, axis=1))
    # print(w_samples)
    # start_time = time.time()
    # l_samples = ep_l(traj_embeds, w_samples[50], latent_dim, num_l_samples)
    # print(time.time() - start_time) # 