from plot import plot_perf
from environment import SimpleSBM, SBM, SimpleCLM
from agent import DUCB, DTS
from time import time
import numpy as np


# save_dir: directory in which will be saved figures
# rm_mode: whether to raise an exception (False) or erase already existing figures (True)
def test_graph(graph, alpha, T, n_rep, alg_name, save_dir=None, rm_mode=False,
               observe_full=True, prior_k=None, prior_beta=None):
    algs = []
    for i in range(n_rep):
        print('Iteration {}/{}'.format(i+1, n_rep),)
        if alg_name == 'DTS':
            alg_ins = DTS(graph, alpha, T, observe_full=observe_full, prior_k=prior_k, prior_beta=prior_beta)
        elif alg_name == 'DUCB':
            alg_ins = DUCB(graph, alpha, T, observe_full=observe_full)
        else:
            raise ValueError(alg_name)
        alg_ins.act()
        algs.append(alg_ins)
    plot_perf(algs, save_dir, rm_mode)
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


def make_prior(exp, var, n):
    b = exp/var
    k = exp*b
    return np.ones(n)*k, np.ones(n)*b

# graph = SimpleSBM(0.005, 0.1, [5,5,10,5,5])
# graph = SimpleSBM(0.025, 0.1, [10, 20, 15, 30])
# graph = SimpleCLM([0.8]*10+[0.1]*100)
# graph = SimpleSBM(0.0001, 0.1, [5]*19 + [20])
#init_time = time()
#test_graph(graph, 0.3, 1000, 10, DUCB, 'results/SBM28_SBM_DUCB', rm_mode=True)
#print('Time1: ', time() - init_time)


# init_time = time()
# test_graph(graph, 0.3, 1000, 4, DUCB, 'results/SBM28_SBM_DUCB', rm_mode=True, observe_full=False)
# print('Time1: ', time() - init_time)
# init_time = time()
# test_graph(graph, 0.3, 1000, 10, DTS, 'results/SBM28_SBM_DTS', rm_mode=True, observe_full=False)
# print('Time2: ', time() - init_time)
smooth_clm = SimpleCLM(np.linspace(0.2,0.8, 100))
vhard_clm = SimpleCLM([0.46]*25 + [0.48]*25 + [0.50]*25 + [0.52]*25)
init_time = time()
k, b = make_prior(50, 10, 100)
test_graph(vhard_clm, 0.3, 3000, 10, 'DTS', 'results/QUICK_VHARD_CLM_DTS', rm_mode=True,
           observe_full=False, prior_k=k, prior_beta=b)
test_graph(vhard_clm, 0.3, 3000, 10, 'DUCB', 'results/QUICK_VHARD_CLM_DUCB', rm_mode=True,
           observe_full=False, prior_k=k, prior_beta=b)
# test_graph(smooth_clm, 0.3, 1000, 3, DUCB, 'results/SMOOTH_CLM_DUCB', rm_mode=True, observe_full=False)
# test_graph(graph, 0.3, 1000, 10, DUCB, 'results/SBM28_CLM_DUCB', rm_mode=True, observe_full=False)
# print('Time3: ', time() - init_time)
# init_time = time()
# test_graph(graph, 0.3, 1000, 10, DTS, 'results/SBM28_CLM_DTS', rm_mode=True, observe_full=False)
# print('Time4: ', time() - init_time)



