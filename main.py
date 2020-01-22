from plot import plot_perf
from environment import SimpleSBM, SimpleCLM
from agent import DUCB, DTS
import numpy as np


# Main function that executes an algorithm on a graph and save/plot the results
# Parameters:
# - alpha and T as in Lugosi et al. 19
# - n_rep is the number of repetitions to get a std
# - alg_name is a string indicating which algorithm to use
# - save_dir: directory in which will be saved figures
# - rm_mode: whether to raise an exception (False) or erase already existing figures (True)
# - observe_full: whether to observe the connected components (slow)
# - priors for the gamma prior of DTS
# - double: parameters of the double function, or None if not used.
def test_graph(graph, alpha, T, n_rep, alg_name, save_dir=None, rm_mode=False,
               observe_full=True, prior_k=None, prior_beta=None, double=None):
    algs = []
    for i in range(n_rep):
        print('Iteration {}/{}'.format(i+1, n_rep))

        # Creating the algorithm object
        if alg_name == 'DTS':
            alg_ins = DTS(graph, alpha, T, observe_full=observe_full, prior_k=prior_k, prior_beta=prior_beta)
        elif alg_name == 'DUCB':
            alg_ins = DUCB(graph, alpha, T, observe_full=observe_full)
        else:
            raise ValueError(alg_name)

        # Double algorithm or not
        if double is None or alg_name != 'DUCB':
            alg_ins.act()
        else:
            alg_ins.act_double(*double)
        algs.append(alg_ins)

    # Plotting/Saving the performances
    plot_perf(algs, save_dir, rm_mode)

    # Saving the parameters of the last algorithm in a text file
    last_alg = algs[-1]
    if save_dir is not None:
        with open(save_dir + '/' + 'parameters.txt', 'w') as f:
            config = 'alpha: {}\n' \
                     'alpha_lim: {}\n' \
                     'T: {}\n' \
                     'n_rep: {}\n' \
                     'mu*: {}\n' \
                     'alpha_mu*: {}\n' \
                     'sample_size: {}'.format(alpha, last_alg.alpha_lim, T, n_rep, last_alg.mu_star,
                                              last_alg.alpha_mu_star, len(last_alg.V0))
            f.write(str(graph)+'\n'+config)


# Outputs Gamma parameters in to have a prior of expectation exp and variance var
def make_prior_gamma(exp, var, n):
    b = exp/var
    k = exp*b
    return np.ones(n)*k, np.ones(n)*b


# Outputs Beta parameters
def make_prior_beta(a,b, n):
    return np.ones((n,n)) * a, np.ones((n,n)) * b


# 1' test on a CLM and a SBM with DTS and DUCB with observe_full=true (slower but more information)
# DTS is run with good priors, you can change the parameters of the prior to see when it fails.
# saves result in current directory
if __name__ == '__main__':
    sbm = SimpleSBM(0.01, 0.1, [5]*5 + [10])
    test_graph(sbm, 0.3, 3000, 3, 'DUCB', 'DUCB_SBM', rm_mode=True, observe_full=True)
    k, beta = make_prior_gamma(5, 2, sbm.n)  # 'Good prior'
    test_graph(sbm, 0.3, 3000, 3, 'DTS', 'DTS_SBM', rm_mode=True, prior_beta=beta, prior_k=k, observe_full=True)

    clm = SimpleCLM(np.linspace(0.2,0.8,50))
    test_graph(clm, 0.3, 3000, 3, 'DUCB', 'DUCB_CLM', rm_mode=True, observe_full=True)
    k, beta = make_prior_gamma(30, 5, clm.n)  # Good prior
    test_graph(clm, 0.3, 3000, 3, 'DTS', 'DTS_CLM', rm_mode=True, prior_beta=beta, prior_k=k, observe_full=True)



