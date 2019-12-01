import numpy as np
from environment import SimpleSBM, SBM
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
            spl_size = min(int(np.ceil(np.log(T)/np.log(1/(1-alpha)))), graph.n)
        print('Sample size', spl_size)
        self.V0 = np.random.choice(range(graph.n), spl_size, False)
        self.mu = np.array([self.graph.observe_degree(i) for i in self.V0], dtype=np.float)
        self.N = np.ones(spl_size, dtype=int)
        self.t = spl_size

        # For logging
        self.cum_reward = 0
        self.cum_degree = 0
        self.opt_cum_reward = 0
        self.alpha_opt_cum_reward = 0
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


    def act(self):
        while self.t < self.T:
            a = self.select_action()
            d, c_list, n_comp = self.graph.observe_full(self.V0[a])
            self.mu[a] = (self.N[a]*self.mu[a] + d)/(self.N[a]+1)
            self.N[a] += 1
            self.t += 1

            # if max(c_list) != int(np.quantile(c_list, 1-self.alpha)):
            #     print('Different !')
            self.cum_degree += d
            self.cum_reward += c_list[a]
            self.opt_cum_reward += max(c_list)
            self.alpha_opt_cum_reward += np.quantile(c_list, 1-self.alpha)
            self.n_comps.append(n_comp)

            self.log()

    def log(self):
        self.degree_regret.append(self.t*self.mu_star - self.cum_degree)
        self.alpha_degree_regret.append(self.t*self.alpha_mu_star - self.cum_degree)
        self.real_regret.append(self.opt_cum_reward - self.cum_reward)
        self.alpha_real_regret.append(self.alpha_opt_cum_reward - self.cum_reward)

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
        # print('U_vect')
        # print(U_vect)
        return np.argmax(U_vect)

    def plot_perf(self):
        plt.hist(self.n_comps)
        plt.title('Histogram of number of connected components')
        plt.show()
        f, axs = plt.subplots(2,2)
        xs = range(len(self.degree_regret))
        axs[0][0].plot(xs, self.degree_regret, label='d')
        axs[0][0].set_title('Degree Regret')
        axs[0][1].plot(xs, self.alpha_degree_regret, label='ad')
        axs[0][1].set_title('Alpha Degree Regret')
        axs[1][0].plot(xs, self.real_regret, label='c')
        axs[1][0].set_title('Component Regret')
        axs[1][1].plot(xs, self.alpha_real_regret, label='ac')
        axs[1][1].set_title('Alpha Component Regret')
        plt.show()


def dicho(f, a, b, eps=1e-6, tmax=1000):
    for i in range(tmax):
        if abs(a-b) < eps:
            return a
        if f((a+b)/2) > 0:
            b = (a+b)/2
        else:
            a = (a+b)/2
    print('Dichotomy failed')
    return a


graph = SimpleSBM(0.005, 0.1, [5,5,10,5,5])
ducb = DUCB(graph, 0.2, 1000)
ducb.act()
ducb.plot_perf()