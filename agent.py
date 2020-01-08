import math
import random
from scipy.stats import gamma

import numpy as np

from barriermethod import barr_method

class DTS:
    def __init__(self, graph, alpha, T, prior_k=None, prior_theta=None, observe_full=True):
        self.graph = graph
        self.alpha = alpha
        self.T = T
        self.alpha_lim = 1 - np.exp(-np.log(T) / np.log(graph.n))
        self.observe_full = observe_full

        if not alpha:
            spl_size = graph.n
        else:
            candidate = np.ceil(np.log(T)/np.log(1/(1-alpha)))
            if candidate > graph.n:
                print('Alpha gave too many nodes for this T, taking graph.n')
            spl_size = min(int(candidate), graph.n)
        self.V0 = np.random.choice(range(graph.n), spl_size, False)
        self.N = np.ones(spl_size, dtype=int)
        self.t = 0

        if prior_k is None:
            self.gamma_k = np.ones(spl_size, dtype=np.float)
        else:
            self.gamma_k = prior_k

        if prior_theta is None:
            self.prior_theta = np.ones(spl_size, dtype=np.float)
            self.gamma_theta = np.ones(spl_size, dtype=np.float)
        else:
            self.prior_theta = prior_theta.copy()
            self.gamma_theta = prior_theta

        # For logging
        self.cum_reward = 0
        self.cum_degree = 0
        self.opt_cum_reward = 0
        self.alpha_opt_cum_reward = 0
        self.a_star = np.argmax(self.graph.P.sum(axis=0))
        self.a_alpha_star = np.argsort(self.graph.P.sum(axis=0))[int((1 - alpha) * graph.P.shape[0])]
        self.mu_star = self.graph.P.sum(axis=0)[self.a_star]
        self.alpha_mu_star = self.graph.P.sum(axis=0)[self.a_alpha_star]

        # List of values:
        self.degree_regret = []
        self.alpha_degree_regret = []
        self.real_regret = []
        self.alpha_real_regret = []
        self.n_comps = []
        self.degrees = []

    def update_belief(self, a, d):
        self.gamma_k[a] += d
        self.gamma_theta[a] = self.prior_theta[a]/(self.N[a]*self.prior_theta[a]+1)

    def act(self):
        while self.t < self.T:
            a = self.select_action()

            if self.observe_full:
                d, c_list, n_comp = self.graph.observe_full(self.V0[a])
            else:
                d, c_list, n_comp = self.graph.observe_degree(self.V0[a]), 0, 0

            self.N[a] += 1
            self.t += 1
            self.update_belief(a, d)

            self.cum_degree += d
            if self.observe_full:
                self.cum_reward += c_list[self.V0[a]]
                self.opt_cum_reward += c_list[self.a_star]
                self.alpha_opt_cum_reward += c_list[self.a_alpha_star]
            else:
                self.cum_reward += 0
                self.opt_cum_reward += 0
                self.alpha_opt_cum_reward += 0
            self.n_comps.append(n_comp)
            self.degrees.append(d)

            self.log()

    def log(self):
        self.degree_regret.append(self.t*self.mu_star - self.cum_degree)
        self.alpha_degree_regret.append(self.t*self.alpha_mu_star - self.cum_degree)
        self.real_regret.append(self.opt_cum_reward - self.cum_reward)
        self.alpha_real_regret.append(self.alpha_opt_cum_reward - self.cum_reward)

    def get_N(self):
        N = np.zeros(self.graph.n, dtype=int)
        N[self.V0] += self.N
        return N

    def select_action(self):
        B = np.array([gamma.rvs(a=k, scale=theta) for k, theta in zip(self.gamma_k, self.gamma_theta)])
        best = np.argwhere(B == np.amax(B))[0]
        return random.choice(best)


# Attention a ne pas confondre a et V0[a] !
class DUCB:
    def __init__(self, graph, alpha, T, observe_full=True):
        self.graph = graph
        self.alpha = alpha
        self.T = T
        self.alpha_lim = 1 - np.exp(-np.log(T) / np.log(graph.n))
        self.observe_full = observe_full

        if not alpha:
            spl_size = graph.n
        else:
            candidate = np.ceil(np.log(T)/np.log(1/(1-alpha)))
            if candidate > graph.n:
                print('Alpha gave too many nodes for this T, taking graph.n')
            spl_size = min(int(candidate), graph.n)
        self.V0 = np.random.choice(range(graph.n), spl_size, False)
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
        self.mu_star = self.graph.P.sum(axis=0)[self.a_star]
        self.alpha_mu_star = self.graph.P.sum(axis=0)[self.a_alpha_star]

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

            if self.observe_full:
                d, c_list, n_comp = self.graph.observe_full(self.V0[a])
            else:
                d, c_list, n_comp = self.graph.observe_degree(self.V0[a]), 0, 0
            self.mu[a] = (self.N[a]*self.mu[a] + d)/(self.N[a]+1)
            self.N[a] += 1
            self.t += 1

            self.cum_degree += d
            if self.observe_full:
                self.cum_reward += c_list[self.V0[a]]
                self.opt_cum_reward += c_list[self.a_star]
                self.alpha_opt_cum_reward += c_list[self.a_alpha_star]
            else:
                self.cum_reward += 0
                self.opt_cum_reward += 0
                self.alpha_opt_cum_reward += 0
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
                spl_size = self.graph.n
            elif self.N.shape[0] >= self.graph.n and not self.V0 is None:
                spl_size = -1
            else:
                candidate = np.ceil(np.log(beta)/np.log(1/(1-self.alpha)))
                if candidate > self.graph.n - self.N.shape[0]:
                    print('Alpha gave too many nodes for this T, taking graph.n')
                spl_size = min(int(candidate), self.graph.n - self.N.shape[0])
            print('Sample size', spl_size)

            if self.V0 is None and spl_size != -1:
                self.V0 = np.random.choice(range(self.graph.n), spl_size, False)
                self.mu = np.array([self.graph.observe_degree(i) for i in self.V0], dtype=np.float)
                self.N  = np.ones(spl_size, dtype=int)
                self.t = spl_size
                self.T *= beta
            elif spl_size != -1:
                untouched = [i for i in list(range(self.graph.n)) if i not in self.V0]
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
                #U_vect.append(barr_method(mu_a, self.t, N_a, mu_a))
                U_vect.append(dicho(U, mu / 2, mu))
        best = np.argwhere(U_vect == np.amax(U_vect))[0]
        return random.choice(best)


def dicho(f, a, b, eps=0.001, tmax=1000):
    for i in range(tmax):
        if abs(a-b) < eps:
            return a
        if f((a+b)/2) > 0:
            b = (a+b)/2
        else:
            a = (a+b)/2
    print('Dichotomy failed')
    return a