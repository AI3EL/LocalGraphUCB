import matplotlib.pyplot as plt
import numpy as np
import os


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
    plt.title('Histogram of number of connected components')
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

    # Mean performances
    if algs[0].observe_full:
        f, axs = plt.subplots(2, 2)
        attr_list = ['degree_regret', 'alpha_degree_regret', 'real_regret', 'alpha_real_regret']
        names = ['degree', 'alpha_degree', 'real', 'alpha_real']
        ax_list = axs.flatten()
    else:
        f, ax_list = plt.subplots(1, 2)
        attr_list = ['degree_regret', 'alpha_degree_regret']
        names = ['degree', 'alpha_degree']

    data = [np.array([getattr(alg, attr) for alg in algs]) for attr in attr_list]
    for ax, dat, name in zip(ax_list, data, names):
        mean = np.mean(dat, axis=0)
        std = np.std(dat, axis=0)
        xs = range(mean.shape[0])
        ax.plot(xs, mean)
        ax.fill_between(xs, mean - 2 * std, mean + 2 * std, alpha=0.15)
        ax.set_title(name)
    save_or_show('mean_performances', save_dir, rm_mode)

    # Each performances
    if algs[0].observe_full:
        f, axs = plt.subplots(2, 2)
        attr_list = ['degree_regret', 'alpha_degree_regret', 'real_regret', 'alpha_real_regret']
        names = ['degree', 'alpha_degree', 'real', 'alpha_real']
        ax_list = axs.flatten()
    else:
        f, ax_list = plt.subplots(1, 2)
        attr_list = ['degree_regret', 'alpha_degree_regret']
        names = ['degree', 'alpha_degree']

    data = [np.array([getattr(alg, attr) for alg in algs]) for attr in attr_list]
    for ax, dat, name in zip(ax_list, data, names):
        xs = range(dat.shape[1])
        for curv in dat:
            ax.plot(xs, curv)
        ax.set_title(name)
    save_or_show('each_performances', save_dir, rm_mode)

    # Mean estimators:
    if type(algs[0]).__name__ == 'DTS':
        f, ax_list = plt.subplots(2, 1)
        attr_list = ['k', 'beta']
        names = ['K posterior', 'Beta posterior']

    elif type(algs[0]).__name__ == 'DUCB':
        f, ax_list = plt.subplots(1, 1)
        attr_list = ['mu']
        names = ['mu']
        ax_list = [ax_list]

    else:
        raise ValueError('Unknown class type: ' + type(algs[0]).__name__)

    data = np.zeros((len(attr_list), len(algs), algs[0].graph.n))
    count = np.zeros((algs[0].graph.n))
    for alg in algs:
        count[alg.V0] += 1

    for attr, dat in zip(attr_list, data):
        for i,alg in enumerate(algs):
            dat[i, alg.V0] = getattr(alg, attr)

    for ax, dat, name in zip(ax_list, data, names):
        mean = np.sum(dat, axis=0) / count
        std = np.std(dat, axis=0)
        xs = range(mean.shape[0])
        ax.plot(xs, mean)
        ax.fill_between(xs, mean - 2 * std, mean + 2 * std, alpha=0.15)
        ax.set_title(name)
    save_or_show('mean_estimators', save_dir, rm_mode)

    # Each estimators:
    if type(algs[0]).__name__ == 'DTS':
        f, ax_list = plt.subplots(2, 1)
        attr_list = ['k', 'beta']
        names = ['K posterior', 'Beta posterior']

    elif type(algs[0]).__name__ == 'DUCB':
        f, ax_list = plt.subplots(1, 1)
        attr_list = ['mu']
        names = ['mu']
        ax_list = [ax_list]

    else:
        raise ValueError('Unknown class type: ' + type(algs[0]).__name__)

    data = np.zeros((len(attr_list), len(algs), algs[0].graph.n))
    # data[:,:,:] = np.nan

    for attr, dat in zip(attr_list, data):
        for i,alg in enumerate(algs):
            dat[i, alg.V0] = getattr(alg, attr)

    for ax, dat, name in zip(ax_list, data, names):
        xs = range(dat.shape[1])
        for curve in dat:
            ax.plot(xs, curve, '.')
        ax.set_title(name)
    save_or_show('each_estimators', save_dir, rm_mode)

    # Each V0:
    f, axs = plt.subplots(1,2)
    count = np.zeros((len(algs), algs[0].graph.n))
    for i,alg in enumerate(algs):
        count[i, alg.V0] = 1
    for cnt in count:
        axs[0].plot(range(count.shape[1]), cnt, '.')
    axs[1].plot(range(count.shape[1]), count.sum(axis=0), '.')
    save_or_show('count', save_dir, rm_mode)

    # Prior:
    if type(algs[0]).__name__ == 'DTS':
        from scipy.stats import gamma
        k, b = algs[0].prior_k, algs[0].prior_beta
        xs = np.linspace(k/b-1, k/b+1, 1000)
        f, ax = plt.subplots(1,1)
        ax.plot(xs, gamma.pdf(xs, a=k, scale=1/b))
        save_or_show('prior', save_dir, rm_mode)

        k, b = algs[0].k, algs[0].beta
        xs = np.linspace(k / b - 1, k / b + 1, 1000)
        f, ax = plt.subplots(1, 1)
        ax.plot(xs, gamma.pdf(xs, a=k, scale=1 / b))
        save_or_show('posterior', save_dir, rm_mode)
