import numpy as np
import matplotlib.pyplot as plt
import os

from environment import SimpleSBM, SBM, SimpleCLM
from agent import DUCB, DTS
from time import time


def plot_perf(algs, save_dir=None, rm_mode=False):
    if save_dir is not None:
        try:
            os.mkdir(save_dir)
        except OSError:
            pass

    def save_or_show(name, save_dir, rm_mode):
        if save_dir is None:
            plt.show()
        else:
            path = save_dir + '/' + name + '.png'
            if not rm_mode and os.path.exists(path):
                raise ValueError('Directory already contains files, aborting')
            else:
                plt.savefig(path)
                plt.figure()

    # Histogram of number of connected components
    hists = [np.histogram(alg.n_comps)[0] for alg in algs]
    bin_edges = [np.histogram(alg.n_comps)[1][:-1] for alg in algs]
    mean = np.mean(hists, axis=0)
    std = np.std(hists, axis=0)
    xs = np.mean(bin_edges, axis=0)
    plt.plot(xs, mean)
    plt.fill_between(xs, mean - 2 * std, mean + 2 * std, alpha=0.15)
    plt.title('Histogram of number of connected components (max={}'.format(algs[0].graph.n))
    save_or_show('component_sizes_histogram', save_dir, rm_mode)

    # Histogram of number of times action was taken
    Ns = np.array([alg.get_N() for alg in algs])
    mean = np.mean(Ns, axis=0)
    std = np.std(Ns, axis=0)
    xs = range(mean.shape[0])
    plt.plot(xs, mean)
    plt.fill_between(xs, mean - 2 * std, mean + 2 * std, alpha=0.15)
    plt.title('Histogram of number of times action was taken')
    save_or_show('N_histogram', save_dir, rm_mode)

    # Performances
    f, axs = plt.subplots(2, 2)
    attr_list = ['degree_regret', 'alpha_degree_regret', 'real_regret', 'alpha_real_regret']
    data = [np.array([getattr(alg, attr) for alg in algs]) for attr in attr_list]
    names = ['degree', 'alpha_degree', 'real', 'alpha_real']
    ax_list = axs.flatten()
    for ax, dat, name in zip(ax_list, data, names):
        mean = np.mean(dat, axis=0)
        std = np.std(dat, axis=0)
        xs = range(mean.shape[0])
        ax.plot(xs, mean)
        ax.fill_between(xs, mean - 2 * std, mean + 2 * std, alpha=0.15)
        ax.set_title(name)
    save_or_show('performances', save_dir, rm_mode)


# save_dir: directory in which will be saved figures
# rm_mode: whether to raise an exception (False) or erase already existing figures (True)
def test_graph(graph, alpha, T, n_rep, alg, save_dir=None, rm_mode=False, observe_full=True):
    algs = []
    for i in range(n_rep):
        print('Iteration {}/{}'.format(i+1, n_rep),)
        alg_ins = alg(graph, alpha, T, observe_full=observe_full)
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


# graph = SimpleSBM(0.005, 0.1, [5,5,10,5,5])
# graph = SimpleSBM(0.025, 0.1, [10, 20, 15, 30])
# graph = SimpleCLM([0.8]*10+[0.1]*100)
init_time = time()
graph = SimpleSBM(0.0001, 0.1, [5]*19 + [20])
graph = SimpleSBM(0.0001, 0.1, [5]*1900 + [20]*100)
#init_time = time()
#test_graph(graph, 0.3, 1000, 10, DUCB, 'results/SBM28_SBM_DUCB', rm_mode=True)
#print('Time1: ', time() - init_time)
init_time = time()
test_graph(graph, 0.3, 1000, 10, DUCB, 'results/SBM28_SBM_DUCB', rm_mode=True, observe_full=False)
print('Time1: ', time() - init_time)
init_time = time()
test_graph(graph, 0.3, 1000, 10, DTS, 'results/SBM28_SBM_DTS', rm_mode=True, observe_full=False)
print('Time2: ', time() - init_time)
graph = SimpleCLM([0.8]*10+[0.1]*100)
init_time = time()
test_graph(graph, 0.3, 1000, 10, DUCB, 'results/SBM28_CLM_DUCB', rm_mode=True, observe_full=False)
print('Time3: ', time() - init_time)
init_time = time()
test_graph(graph, 0.3, 1000, 10, DTS, 'results/SBM28_CLM_DTS', rm_mode=True, observe_full=False)
print('Time4: ', time() - init_time)
