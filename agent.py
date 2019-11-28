import numpy as np
from environment import SimpleSBM, SBM

# TODO: approximer les ci pour estimer le regret
# TODO: implementer DUCB double$$

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
        if not alpha:
            spl_size = graph.n
        else:
            spl_size = np.ceil(np.log(T)/np.log(1/(1-alpha)))

        self.V0 = np.random.choice(range(graph.n), spl_size, False)
        self.mu = np.array([self.graph.observe_degree(i) for i in self.V0], dtype=np.float)
        self.N = np.ones(spl_size, dtype=int)
        self.t = spl_size

        # For logging
        self.cum_reward = 0
        self.cum_degree = 0
        self.mu_star = max(self.graph.P.sum(axis=0))
        self.alpha_mu_star = np.quantile(self.graph.P.sum(axis=0), 1-alpha)

    def act(self):
        while self.t < self.T:
            a = self.select_action()
            d, mu = self.graph.observe_full(self.V0[a])
            self.mu[a] = (self.N[a]*self.mu[a] + d)/(self.N[a]+1)
            self.N[a] += 1
            self.t += 1
            self.cum_degree += d
            self.cum_reward += mu

    def log(self):
        degree_regret = self.t*self.mu_star - self.cum_degree
        alpha_degree_regret = self.t*self.alpha_mu_star - self.cum_degree
        print('Degree regret :', degree_regret)
        print('Alpha degree regret :', alpha_degree_regret)

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


def dicho(f, a, b, eps=1e-6, tmax=100):
    for i in range(tmax):
        if abs(a-b) < eps:
            return a
        if f((a+b)/2) > 0:
            b = (a+b)/2
        else:
            a = (a+b)/2
    print('Dichotomy failed')
    return a

graph = SimpleSBM(0.05, 0.2, [2,8])
ducb = DUCB(graph, 0, 1000)
ducb.act()
ducb.log()