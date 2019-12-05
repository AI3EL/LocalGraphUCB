import numpy as np
from environment import SimpleSBM, SBM, SimpleCLM
import matplotlib.pyplot as plt

# TODO: PB: l'estimation du alpha regret est lineaire ...

# TODO: trouver comment choisir les parametres de graphe pour ne pas avoir une composante connexe uniquement
# TODO: ou n composante connexes ... : un truc du genre p = 1/n pour l'exterieur et autre pour les diffs.
# TODO: implementer DUCB double
# TODO: implementer Wung-Chu

# TODO: discussion avec Claire:
# TODO - Reflechir a pourquoi RT_alpha plutot que juste RT
# TODO - Pas de borne inferieure du regret dans le papier, alors que c'est usuel, pourquoi ?
# TODO - Vérifier ce que le papier dit a propos de n pas trop grand.
# TODO - Trouver des Kij qui ne verifient pas l'hypothese 1 et verifier/comprendre que ca marche aps

# Attention a ne pas confondre a et V0[a] !
class DUCB:
    def __init__(self, graph, alpha, T):
        self.graph = graph
        self.alpha = alpha
        self.T = T
        print('For this T, alpha_lim: ', 1 - np.exp(-np.log(T) / np.log(graph.n)))

        if not alpha:
            spl_size = graph.n
        else:
            candidate = np.ceil(np.log(T)/np.log(1/(1-alpha)))
            if candidate > graph.n:
                print('Alpha gave too many nodes for this T, taking graph.n')
            spl_size = min(int(candidate), graph.n)
        print('Sample size', spl_size)
        self.V0 = np.random.choice(range(graph.n), spl_size, False)
        self.V0.sort()  # TODO: remove
        print(self.V0)
        self.mu = np.array([self.graph.observe_degree(i) for i in self.V0], dtype=np.float)
        self.N = np.ones(spl_size, dtype=int)
        self.t = spl_size

        # For logging
        self.cum_reward = 0
        self.cum_degree = 0
        self.opt_cum_reward = 0
        self.alpha_opt_cum_reward = 0
        self.a_star = np.argmax(self.graph.P.sum(axis=0))
        self.a_alpha_star = np.argsort(self.graph.P.sum(axis=0))[int((1-alpha)*graph.P.shape[0])]
        self.mu_star = max(self.graph.P.sum(axis=0))
        self.alpha_mu_star = np.quantile(self.graph.P.sum(axis=0), 1-alpha)
        print('mustar',self.mu_star)
        print('alpha_mustar',self.alpha_mu_star)

        # List of values:
        self.degree_regret = []
        self.alpha_degree_regret = []
        self.real_regret = []
        self.alpha_real_regret = []
        self.n_comps = []
        self.degrees = []

    def act(self):
        while self.t < self.T:
            a = self.select_action()
            d, c_list, n_comp = self.graph.observe_full(self.V0[a])
            self.mu[a] = (self.N[a]*self.mu[a] + d)/(self.N[a]+1)
            self.N[a] += 1
            self.t += 1

            self.cum_degree += d
            self.cum_reward += c_list[a]
            self.opt_cum_reward += c_list[self.a_star]
            self.alpha_opt_cum_reward += c_list[self.a_alpha_star]
            self.n_comps.append(n_comp)
            self.degrees.append(d)

            self.log()

    def act_double(self, k_max, beta):
        self.V0 = None
        self.T = 1
        self.t = 0
        end = False
        for k in range(k_max):
            if not self.alpha and not self.V0 is None:
                spl_size = graph.n
            elif self.N.shape[0] >= graph.n and not self.V0 is None:
                spl_size = -1
            else:
                candidate = np.ceil(np.log(beta)/np.log(1/(1-self.alpha)))
                if candidate > graph.n - self.N.shape[0]:
                    print('Alpha gave too many nodes for this T, taking graph.n')
                spl_size = min(int(candidate), graph.n - self.N.shape[0])
            print('Sample size', spl_size)

            if self.V0 is None and spl_size != -1:
                self.V0 = np.random.choice(range(graph.n), spl_size, False)
                self.mu = np.array([self.graph.observe_degree(i) for i in self.V0], dtype=np.float)
                self.N  = np.ones(spl_size, dtype=int)
                self.t = spl_size
                self.T *= beta
            elif spl_size != -1:
                untouched = [i for i in list(range(graph.n)) if i not in self.V0]
                Uk = np.random.choice(untouched, spl_size, False)
                self.V0 = np.concatenate((self.V0, Uk))
                self.mu = np.concatenate((self.mu,
                                         np.array([self.graph.observe_degree(i) for i in Uk], dtype=np.float)))
                self.N = np.concatenate((self.N, np.ones(spl_size, dtype=int)))
                self.T *= beta
            else:
                print("V0 full")
                self.T = pow(beta, k_max)
                end = True

            self.act()

            if end: break



    def log(self):
        self.degree_regret.append(self.t*self.mu_star - self.cum_degree)
        self.alpha_degree_regret.append(self.t*self.alpha_mu_star - self.cum_degree)
        self.real_regret.append(self.opt_cum_reward - self.cum_reward)
        self.alpha_real_regret.append(self.alpha_opt_cum_reward - self.cum_reward)

    def get_N(self):
        N = np.zeros(self.graph.n, dtype=int)
        N[self.V0] += self.N
        return N

    # Etude derivee montre que f en V, min en mu_a. Donc mu* >= mua
    def select_action(self):
        log_mu = np.log(self.mu)
        U_vect = []
        for mu_a, log_mu_a, N_a in zip(self.mu, log_mu, self.N):
            U = lambda mu: mu + mu_a * (log_mu_a - np.log(mu) - 1) - 3 * np.log(self.t) / N_a
            mu = mu_a

            # Ill defined but quite intuitive
            if not mu:
                U_vect.append(np.inf)
            # Convention de sup ensemble vide = -inf
            elif U(mu) > 0:
                U_vect.append(-np.inf)
            else:
                while U(mu) < 0:
                    mu *= 2
                U_vect.append(dicho(U, mu / 2, mu))
                assert U(U_vect[-1]) < 0
        # print('mu')
        # print(self.mu)
        # print('U_vect')
        # print(U_vect)
        return np.argmax(U_vect)

    @staticmethod
    def plot_perf(algs):
        # plt.hist(self.n_comps)
        # plt.title('Histogram of number of connected components')
        # plt.show()
        hists = [np.histogram(alg.n_comps)[0] for alg in algs]
        bin_edges = [np.histogram(alg.n_comps)[1][:-1] for alg in algs]
        mean = np.mean(hists, axis=0)
        std = np.std(hists, axis=0)
        xs = np.mean(bin_edges, axis=0).astype(int)
        plt.plot(xs, mean)
        plt.fill_between(xs, mean - 2 * std, mean + 2 * std, alpha=0.15)
        plt.title('Histogram of number of connected components')
        plt.show()

        plt.figure()
        Ns = np.array([alg.get_N() for alg in algs])
        mean = np.mean(Ns, axis=0)
        std = np.std(Ns, axis=0)
        xs = range(mean.shape[0])
        plt.plot(xs, mean)
        plt.fill_between(xs, mean - 2 * std, mean + 2 * std, alpha=0.15)
        plt.title('Histogram of number of times action was taken')
        plt.show()

        f, axs = plt.subplots(2,2)
        attr_list = ['degree_regret', 'alpha_degree_regret', 'real_regret', 'alpha_real_regret']
        data = [np.array([getattr(alg, attr) for alg in algs]) for attr in attr_list]
        names = ['degree', 'alpha_degree', 'real', 'alpha_real']
        ax_list = axs.flatten()
        for ax, dat, name in zip(ax_list, data, names):
            mean = np.mean(dat, axis=0)
            std = np.std(dat, axis=0)
            xs = range(mean.shape[0])
            ax.plot(xs, mean)
            ax.fill_between(xs, mean - 2*std, mean+2*std, alpha=0.15)
            ax.set_title(name)
        plt.show()


def dicho(f, a, b, eps=1e-3, tmax=1000):
    for i in range(tmax):
        if abs(a-b) < eps:
            return a
        if f((a+b)/2) > 0:
            b = (a+b)/2
        else:
            a = (a+b)/2
    print('Dichotomy failed')
    return a


#graph = SimpleSBM(0.005, 0.1, [5,5,10,5,5])
# graph = SimpleSBM(0.025, 0.1, [10, 20, 15, 30])
#graph = SimpleCLM([0.8]*10+[0.1]*100)
graph = SimpleSBM(0.001, 0.01, [50,50,100,50]) #test1, 0.01 0.1  50 50 100 50        double k 10 beta 2
n_rep = 3
ducbs = []
for i in range(n_rep):
    ducb = DUCB(graph, 0.2, 1000)
    #ducb.act()
    ducb.act_double(k_max=20, beta=1.5)
    #ducb.act_double(k_max=75, beta=1.1)
    ducbs.append(ducb)

DUCB.plot_perf(ducbs)