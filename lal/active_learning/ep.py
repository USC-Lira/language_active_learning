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

    traj_embeds = traj_embeds[1:]
    feedback_embeds = feedback_embeds[1:]
    # traj_embeds = traj_embeds[-1:]
    # feedback_embeds = feedback_embeds[-1:]

    # tmp_traj = []
    # tmp_feedback = []
    # for idx in range(traj_embeds.shape[0]):
        # print(np.linalg.norm(feedback_embeds[idx]), np.linalg.norm(traj_embeds[idx]))
        # elt = feedback_embeds[idx].reshape(latent_dim, 1) @ feedback_embeds[idx].reshape(1, latent_dim)
        # print(np.linalg.norm(elt), np.std(elt), np.sum(elt))
        # if np.sum(elt) >= 1:
    #     if idx != 3:
    #     # if abs(feedback_embeds[idx] @ traj_embeds[idx]) < 3: 
    #         tmp_traj.append(traj_embeds[idx])
    #         tmp_feedback.append(feedback_embeds[idx])

    # tmp_traj = np.array(tmp_traj)
    # tmp_feedback = np.array(tmp_feedback)
    # traj_embeds = tmp_traj
    # feedback_embeds = tmp_feedback

    def exponential_decay_damping(alpha_0, K, min_alpha):
        """
        Compute the exponentially decreasing damping factors.
        
        Parameters:
        iterations (int): Total number of iterations.
        K (int): Iteration at which 90% decay is achieved.
        
        Returns:
        numpy.ndarray: Array of damping factors for each iteration.
        """
        alpha_min = min(1 / K, min_alpha)
        decay_rate = np.log(alpha_0 / alpha_min) / (0.9 * K)
        damping_factors = alpha_0 * np.exp(-decay_rate * np.arange(K))
        return damping_factors

    def f(x, mu, cov, psi, tau):
        # w = psi.reshape(-1, 1) @ np.array([[x]])
        # if np.linalg.norm(psi.reshape(-1, 1) @ np.array([[x]])) > 1: return 0
        return scipy.stats.norm.pdf(x, mu, cov) * np.exp(x - psi @ tau)
        # return 100*scipy.stats.norm.pdf(x, mu, cov) * np.exp(x - psi @ tau)

        # if abs(x) > 2: return 0
        # return scipy.stats.norm.pdf(x, mu, cov) * np.exp(x)

    # def f(x, mu, cov):
    #     return scipy.stats.norm.pdf(x, mu, cov) * np.exp(x)

    # following https://www.jmlr.org/papers/volume24/23-0104/23-0104.pdf page 7, algorithm 1
    # https://www.youtube.com/watch?v=0tomU1q3AdY resource

    num_iterations = 2
    # num_iterations = traj_embeds.shape[0]+1
    eta_0 = 0.5 # step size
    eta = 0.5
    alpha_0 = 0.5
    min_alpha = 0.5
    min_eta = 0.01
    alphas = [min_alpha] + [min_alpha] * (traj_embeds.shape[0] - 1)
    num_queries = traj_embeds.shape[0]
    # num_queries = min(3, traj_embeds.shape[0])
    # alphas = exponential_decay_damping(alpha_0, traj_embeds.shape[0], min_alpha)
    # etas = exponential_decay_damping(eta_0, traj_embeds.shape[0], min_eta)
    max_Q_delta = 0
    max_r_delta = 0
    Q_k = [np.eye(1) for q in range(num_queries)]
    r_k = [np.zeros(1) for q in range(num_queries)]
    Q_dot = np.sum([feedback_embeds[q].reshape(latent_dim, 1) @ Q_k[q] @ feedback_embeds[q].reshape(1, latent_dim) for q in range(num_queries)], axis=0) # (512, 512)
    r_dot = np.sum([feedback_embeds[q].reshape(latent_dim, 1) @ r_k[q] for q in range(num_queries)], axis=0) # (512,)

    # print(np.linalg.norm(feedback_embeds, axis=1))
    # initial_sigma = np.linalg.pinv(Q_dot) # (512, 512)
    # print(f"Norm of initial mu is: {np.linalg.norm(initial_sigma @ r_dot)}, norm of initial sigma is: {np.linalg.norm(initial_sigma)}")
    # curr_deltas = np.array([0] * traj_embeds.shape[0])
    limit = 2
    # print(alphas)

    # for i in range(num_iterations):
    i = 0
    while i < 1:
        Sigma_dot = np.linalg.pinv(Q_dot) # (512, 512)
        mu_dot = Sigma_dot @ r_dot # (512,)
        curr_Q_delta = 0
        curr_r_delta = 0
        alpha = alphas[min(i, traj_embeds.shape[0]-1)]
        # eta = etas[min(i, traj_embeds.shape[0]-1)]
        # if (np.sum(Sigma_dot) < 1): break
        # if (np.linalg.norm(Sigma_dot) < 1.5): break
        for q in range(num_queries):
            # alpha = alphas[q]
            # print(q, feedback_embeds[q] @ traj_embeds[q])
            # if q == 3 or q == 7: continue

            # EP Downdate
            Sigma_star = feedback_embeds[q].reshape(1, latent_dim) @ Sigma_dot @ feedback_embeds[q].reshape(latent_dim, 1) # (1,)
            # print(Sigma_star, np.linalg.inv(Q_k[q]))
            mu_star = feedback_embeds[q].reshape(1, latent_dim) @ mu_dot # (1,)
            Q_star = np.linalg.inv(Sigma_star) # (1,)
            r_star = Q_star @ mu_star # (1,)
            # print(i, q, Q_star, Q_k[q])
            Q_cav = Q_star - eta * Q_k[q]
            r_cav = r_star - eta * r_k[q]
            # Sigma_cav = 1e-8
            # for k in range(10):
            #     if Sigma_cav >= 1e-3: break
            #     eta = eta * 0.9
            #     Q_cav = Q_star - eta * Q_k[q]
            #     Sigma_cav = np.linalg.inv(Q_cav)
            Sigma_cav = np.linalg.inv(Q_cav)
            mu_cav = Sigma_cav @ r_cav

            # Tilted distribution inference
            zero_mom = np.maximum(scipy.integrate.quad(f, -limit, limit, args=(mu_cav.item(), Sigma_cav.item(), feedback_embeds[q], traj_embeds[q]))[0], 1e-8)
            # zero_mom = np.maximum(scipy.integrate.quad(f, -limit, limit, args=(mu_cav.item(), Sigma_cav.item()))[0], 1e-8)
            # if zero_mom == 1e-8: zero_mom = np.exp(mu_cav.item()) # delta fxn approximation
            # print(zero_mom)
            first_fxn = lambda x, mu, cov, psi, tau: f(x, mu, cov, psi, tau) * x
            # first_fxn = lambda x, mu, cov: f(x, mu, cov) * x
            first_mom = scipy.integrate.quad(first_fxn, -limit, limit, args=(mu_cav.item(), Sigma_cav.item(), feedback_embeds[q], traj_embeds[q]))[0]
            # first_mom = scipy.integrate.quad(first_fxn, -limit, limit, args=(mu_cav.item(), Sigma_cav.item()))[0]
            second_fxn = lambda x, mu, cov, psi, tau: f(x, mu, cov, psi, tau) * x**2
            # second_fxn = lambda x, mu, cov: f(x, mu, cov) * x**2
            second_mom = scipy.integrate.quad(second_fxn, -limit, limit, args=(mu_cav.item(), Sigma_cav.item(), feedback_embeds[q], traj_embeds[q]))[0]
            # second_mom = scipy.integrate.quad(second_fxn, -limit, limit, args=(mu_cav.item(), Sigma_cav.item()))[0]
            # if zero_mom == 1e-8:
            #     curr_deltas[q] = 1
            #     # print("delta: SKIPPED")
            #     continue
            Sigma_tilted = np.array(max(second_mom / zero_mom - (first_mom / zero_mom) * (first_mom / zero_mom), 1e-8)).reshape(1, 1)
            mu_tilted = np.array(first_mom / zero_mom).reshape(1,)

            # add noise
            # sigma_noise = np.random.normal(0, 1e-6)
            # mu_noise = np.random.normal(0, 1e-6)
            # Sigma_tilted += sigma_noise
            # mu_tilted += mu_noise
            # print(sigma_noise, mu_noise)

            # EP update
            Q_tilted = np.linalg.inv(Sigma_tilted)
            r_tilted = Q_tilted @ mu_tilted
            Q_tilde = (1 - alpha) * Q_k[q] + (alpha/eta) * (Q_tilted - Q_cav)
            r_tilde = (1 - alpha) * r_k[q] + (alpha/eta) * (r_tilted - r_cav)
            # print(np.sum(feedback_embeds[q].reshape(latent_dim, 1) @ (Q_tilde - Q_k[q]) @ feedback_embeds[q].reshape(1, latent_dim)), np.sum(feedback_embeds[q].reshape(latent_dim, 1) @ (r_tilde - r_k[q])))
            # print(Q_tilde, Q_k[q], r_tilde, r_k[q])
            # print(Q_tilted, Q_cav, r_tilted, r_cav)
            # print(Q_k[q].copy(), Q_tilde.copy(), r_k[q].copy(), r_tilde.copy())
            Q_dot += feedback_embeds[q].reshape(latent_dim, 1) @ (Q_tilde - Q_k[q]) @ feedback_embeds[q].reshape(1, latent_dim)
            r_dot += feedback_embeds[q].reshape(latent_dim, 1) @ (r_tilde - r_k[q])
            # print(np.linalg.norm(np.linalg.pinv(Q_dot)), np.std(np.linalg.pinv(Q_dot)))
            # if q == 3: continue
            Q_k_tmp = Q_k[q].copy()
            r_k_tmp = r_k[q].copy()
            Q_k[q] = Q_tilde.copy()
            r_k[q] = r_tilde.copy()
            # print(Q_tilde, r_tilde)
            # curr_Q_delta = max(curr_Q_delta, np.linalg.norm(Q_k_tmp - Q_k[q]))
            # curr_r_delta = max(curr_r_delta, np.linalg.norm(r_k_tmp - r_k[q]))
            # curr_deltas[q] = 0
            # print("delta: ", i, q, curr_Q_delta, curr_r_delta)
        # Q_dot = (Q_dot + Q_dot.T) / 2 # pretty much the matrix is a transpose of itself, it's just probably not equal due to floating point error
        # min_eig = np.min(np.real(np.linalg.eigvals(Q_dot)))
        # if min_eig < 0:
            # Q_dot -= (min_eig - 1e-8) * np.eye(*Q_dot.shape)
        # r_dot /= np.linalg.norm(r_dot) # this line may help seed 2 but worsen seed 1
        # if i == 0:
        #     max_Q_delta = curr_Q_delta
        #     max_r_delta = curr_r_delta
        # else:
        #     # print(max_Q_delta, curr_Q_delta, max_r_delta, curr_r_delta)
        #     # if np.linalg.norm(np.linalg.norm(np.linalg.pinv(Q_dot) @ r_dot)) <= 1 and np.linalg.norm(np.linalg.pinv(Q_dot)) <= 1:
        #     if np.linalg.norm(np.linalg.norm(np.linalg.pinv(Q_dot) @ r_dot)) <= 1:
        #         print("Converged Cov Norm")
        #         break
        #     # if curr_Q_delta < 0.05 * max_Q_delta and curr_r_delta < 0.05 * max_r_delta: 
        #     #     print("Converged Delta")
        #     #     break
        #     if all(curr_deltas == 1):
        #         print("All Cov Low")
        #         break
        i += 1

    print(f"Took {i+1} iterations")
    final_sigma = np.linalg.pinv(Q_dot)
    # print(f"Final_sigma: {np.linalg.norm(final_sigma)}")
    final_sigma = (final_sigma + final_sigma.T) / 2
    min_eig = np.min(np.real(np.linalg.eigvals(final_sigma)))
    if min_eig < 0:
        final_sigma -= (min_eig - 1e-8) * np.eye(*final_sigma.shape)
    final_mu = final_sigma @ r_dot
    # print(np.sum(final_sigma), np.sum(final_mu))
    # print(f"Final_sigma: {np.linalg.norm(final_sigma)}, Q_dot: {np.linalg.norm(Q_dot)}, r_dot: {np.linalg.norm(r_dot)}")
    final_sigma /= (np.linalg.norm(final_sigma)) / 1e-2
    final_mu /= np.maximum(np.linalg.norm(final_mu), 1e-8)
    # print(f"Norm of mu is: {np.linalg.norm(final_mu)}, norm of sigma is: {np.linalg.norm(final_sigma)}")
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

    def sample_hybrid(mu, cov, tau, psi):
        def constraint_function(w):
            return 1 - np.linalg.norm(w)

        def neg_logprob(w): # the tilted/hybrid distribution
            return -(multivariate_normal_pdf(w, mu, cov) * np.exp((psi @ (w.astype(np.float32)-tau)).item()))

        constraints = {'type': 'ineq', 'fun': constraint_function}
        solution = scipy.optimize.minimize(neg_logprob, np.zeros(latent_dim), method='SLSQP', constraints=constraints) # optimize using SLSQP to get the true mode
        solution = scipy.optimize.minimize(neg_logprob, solution.x, method='L-BFGS-B', options={"maxiter":1}) # optimize using L-BFGS-B to get the approx hessian
        mode = solution.x / np.maximum(np.linalg.norm(solution.x), 1e-8)
        inv_hess = solution.hess_inv.todense() * 1e-2
        return np.nan_to_num(mode), np.nan_to_num(inv_hess)

    traj_embeds = traj_embeds[1:]
    feedback_embeds = feedback_embeds[1:]

    num_iterations = 1
    A = np.stack([np.eye(latent_dim) for q in range(traj_embeds.shape[0])]) # total precision
    r = np.stack([np.zeros(latent_dim) for q in range(traj_embeds.shape[0])]) # total shift
    eta = 0.5
    alpha = 0.5

    for i in range(num_iterations):
        A_clone = np.copy(A)
        r_clone = np.copy(r)
        for q in range(traj_embeds.shape[0]):
            # https://www.youtube.com/watch?v=0tomU1q3AdY resource
            # https://jmlr.org/papers/volume21/18-817/18-817.pdf , page 32
            # natural parameters of cavity distribution
            A_cav = np.sum(A, axis=0) - eta * A[q] # cavity precision
            r_cav = np.sum(r, axis=0) - eta * r[q] # cavity shift
            Sigma_cav = np.linalg.pinv(A_cav) # cavity covariance, which is just the inverse of cavity precision
            mu_cav = Sigma_cav @ r_cav # cavity mu, which is just the covariance * cavity shift

            # get tilted/hybrid
            mu_tilted, cov_tilted = sample_hybrid(mu_cav, Sigma_cav, traj_embeds[q], feedback_embeds[q])

            # change back to natural params and update approx
            A[q] = alpha * ((1 / eta) * (np.linalg.inv(cov_tilted) - A_cav) - A[q]) # new precision;
            r[q] = alpha * ((1 / eta) * (A[q] @ mu_tilted - r_cav) - r[q]) # new shift

    sigma = np.linalg.pinv(np.sum(A_clone, axis=0))
    mu = sigma @ np.sum(r_clone, axis=0)
    w_samples = np.random.multivariate_normal(mu, sigma, num_w_samples)
    return w_samples

def ep_l_dimension(traj_embeds, feedback_embeds, latent_dim, w_samples, num_l_samples):
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
        l_samples = np.random.multivariate_normal(mu, cov, num_l_samples).reshape(num_l_samples, latent_dim)
        return l_samples

    traj_embeds = traj_embeds[1:]

    def exponential_decay_damping(alpha_0, K, min_alpha):
        """
        Compute the exponentially decreasing damping factors.
        
        Parameters:
        iterations (int): Total number of iterations.
        K (int): Iteration at which 90% decay is achieved.
        
        Returns:
        numpy.ndarray: Array of damping factors for each iteration.
        """
        alpha_min = min(1 / K, min_alpha)
        decay_rate = np.log(alpha_0 / alpha_min) / (0.9 * K)
        damping_factors = alpha_0 * np.exp(-decay_rate * np.arange(K))
        return damping_factors

    def f(x, mu, cov):
        return scipy.stats.norm.pdf(x, mu, cov) * np.exp(x)

    num_iterations = 2
    eta = 0.5 # step size
    alpha_0 = 0.5
    alphas = exponential_decay_damping(alpha_0, traj_embeds.shape[0], 0.01)
    max_Q_delta = 0
    max_r_delta = 0
    Q_k = [np.eye(1) for q in range(traj_embeds.shape[0])]
    r_k = [np.zeros(1) for q in range(traj_embeds.shape[0])]
    Q_dot = np.sum([feedback_embeds[q].reshape(latent_dim, 1) @ Q_k[q] @ feedback_embeds[q].reshape(1, latent_dim) for q in range(feedback_embeds.shape[0])], axis=0) # (512, 512)
    r_dot = np.sum([feedback_embeds[q].reshape(latent_dim, 1) @ r_k[q] for q in range(feedback_embeds.shape[0])], axis=0) # (512,)

    for i in range(num_iterations):
        Sigma_dot = np.linalg.pinv(Q_dot) # (512, 512)
        mu_dot = Sigma_dot @ r_dot # (512,)
        curr_Q_delta = 0
        curr_r_delta = 0
        for q in range(traj_embeds.shape[0]):
            alpha = alphas[min(i, traj_embeds.shape[0]-1)]
            # EP Downdate
            Sigma_star = feedback_embeds[q].reshape(1, latent_dim) @ Sigma_dot @ feedback_embeds[q].reshape(latent_dim, 1) # (1,)
            mu_star = feedback_embeds[q].reshape(1, latent_dim) @ mu_dot # (1,)
            Q_star = np.linalg.inv(Sigma_star) # (1,)
            r_star = Q_star @ mu_star # (1,)
            Q_cav = Q_star - eta * Q_k[q]
            r_cav = r_star - eta * r_k[q]
            Sigma_cav = np.linalg.inv(Q_cav)
            mu_cav = Sigma_cav @ r_cav

            # Tilted distribution inference
            zero_mom = np.maximum(scipy.integrate.quad(f, -1, 1, args=(mu_cav.item(), Sigma_cav.item()))[0], 1e-8)
            first_fxn = lambda x, mu, cov: f(x, mu, cov) * x
            first_mom = scipy.integrate.quad(first_fxn, -1, 1, args=(mu_cav.item(), Sigma_cav.item()))[0]
            second_fxn = lambda x, mu, cov: f(x, mu, cov) * x**2
            second_mom = scipy.integrate.quad(second_fxn, -1, 1, args=(mu_cav.item(), Sigma_cav.item()))[0]
            Sigma_tilted = np.array(max(second_mom / zero_mom - (first_mom / zero_mom) * (first_mom / zero_mom), 1e-8)).reshape(1, 1)
            mu_tilted = np.array(first_mom / zero_mom).reshape(1,)

            # EP update
            Q_tilted = np.linalg.inv(Sigma_tilted)
            r_tilted = Q_tilted @ mu_tilted
            # Q_tilde = (1 - alpha) * Q_k[q] + (alpha/eta) * (Q_tilted - Q_cav)
            Q_tilde = max((1 - alpha) * Q_k[q] + (alpha/eta) * (Q_tilted - Q_cav), np.array([[1e-8]]))
            r_tilde = (1 - alpha) * r_k[q] + (alpha/eta) * (r_tilted - r_cav)
            Q_dot += feedback_embeds[q].reshape(latent_dim, 1) @ (Q_tilde - Q_k[q]) @ feedback_embeds[q].reshape(1, latent_dim)
            r_dot += feedback_embeds[q].reshape(latent_dim, 1) @ (r_tilde - r_k[q])
            Q_k_tmp = Q_k[q].copy()
            r_k_tmp = r_k[q].copy()
            Q_k[q] = Q_tilde.copy()
            r_k[q] = r_tilde.copy()
            curr_Q_delta = max(curr_Q_delta, np.linalg.norm(Q_k_tmp - Q_k[q]))
            curr_r_delta = max(curr_r_delta, np.linalg.norm(r_k_tmp - r_k[q]))

        Q_k = Q_k_copy.copy()
        r_k = r_k_copy.copy()
        Q_dot = Q_dot_copy.copy()
        r_dot = r_dot_copy.copy()

    final_sigma = np.linalg.inv(Q_dot)
    final_sigma = (final_sigma + final_sigma.T) / 2
    min_eig = np.min(np.real(np.linalg.eigvals(final_sigma)))
    if min_eig < 0:
        final_sigma -= (min_eig - 1e-8) * np.eye(*final_sigma.shape)
    final_mu = final_sigma @ r_dot
    final_sigma /= (np.linalg.norm(final_sigma))
    final_sigma *= 1e-8
    final_mu /= np.maximum(np.linalg.norm(final_mu), 1e-8)
    l_samples = np.random.multivariate_normal(final_mu, final_sigma, num_l_samples)
    return l_samples

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
    traj_embeds = np.random.randn(5, latent_dim)
    traj_embeds /= np.linalg.norm(traj_embeds)
    feedback_embeds = np.random.randn(5, latent_dim)
    feedback_embeds /= np.linalg.norm(feedback_embeds)
    num_w_samples=100
    num_l_samples=100
    # start_time = time.time()
    # w_samples = ep_w(traj_embeds, feedback_embeds, latent_dim, num_w_samples)
    # print(time.time() - start_time) # 300 seconds using njit, 20 queries, 100 hybrid samples, 5 iterations
    start_time = time.time()
    w_samples = ep_w_dimension(traj_embeds, feedback_embeds, latent_dim, num_w_samples)
    print(f"{time.time() - start_time} seconds") # 300 seconds using njit, 20 queries, 100 hybrid samples, 5 iterations
    print(np.linalg.norm(w_samples, axis=1))
    # print(w_samples)
    # start_time = time.time()
    # l_samples = ep_l(traj_embeds, w_samples[50], latent_dim, num_l_samples)
    # print(time.time() - start_time) # 