import numpy as np


class CC:
    def __init__(self, n):
        self.n = n
        self.comps = list(range(n))
        self.members = [[i] for i in range(n)]

    # put the comp of a in the comp of b
    def merge(self, a, b):

        a_comp = self.comps[a]
        b_comp = self.comps[b]
        if a_comp == b_comp:
            return

        for i in self.members[a_comp]:
            self.comps[i] = b_comp
            self.members[b_comp].append(i)
        self.members[a_comp] = []

    def get_size(self, a):
        return len(self.members[self.comps[a]])


class InfluenceGraph:
    def __init__(self, P):
        assert P.shape[0] == P.shape[1]
        assert (P == P.T).all()
        self.P = P
        self.n = P.shape[0]

    def observe_degree(self, a):
        r = np.random.rand(self.n-1)
        mask = np.ones(self.n, dtype=bool)
        mask[a] = False
        return sum((r < self.P[a][mask]).astype(int))

    def observe_full(self, a):
        d = 0
        components = CC(self.n)
        for i in range(self.n):
            for j in range(i+1, self.n):
                r = np.random.rand(1)[0]
                if self.P[i, j] > r:
                    if i == a or j == a:
                        d += 1
                    components.merge(i, j)
        return components.get_size(a), d


class SBM(InfluenceGraph):
    @staticmethod
    def build_P(K, pops):
        S = K.shape[0]
        n = sum(pops)
        cum_pops = np.concatenate(([0], np.cumsum(pops)))
        P = np.empty((n, n))
        for i in range(S):
            rng_i = range(cum_pops[i], cum_pops[i + 1])
            for j in range(S):
                k = K[i, j]
                rng_j = range(cum_pops[j], cum_pops[j + 1])
                P[np.ix_(rng_i, rng_j)] = k * np.ones((pops[i], pops[j]))
        return P

    def __init__(self, K, pops):
        assert K.shape[0] == len(pops) == K.shape[1]
        self.pops = pops
        self.K = K
        self.S = K.shape[0]
        P = SBM.build_P(K, pops)
        InfluenceGraph.__init__(self, P)


class SimpleSBM(SBM):
    def __init__(self, k_diff, k_same, pops):
        S = len(pops)
        K = np.ones((S,S))*k_diff
        np.fill_diagonal(K, k_same)
        SBM.__init__(self, K, pops)


# Attention a ne pas confondre a et V0[a] !
class DUCB:
    def __init__(self, graph, V0):
        self.env = graph
        self.V0 = V0
        self.mu = np.array([self.env.observe_degree(i) for i in V0], dtype=np.float)
        self.N = np.ones(len(V0), dtype=int)
        self.t = len(V0)
        print('d-UCB created')

    def act(self, T):
        while self.t < T:
            a = self.select_action()
            d = self.env.observe_degree(self.V0[a])
            self.mu[a] = (self.N[a]*self.mu[a] + d)/(self.N[a]+1)
            self.N[a] += 1
            self.t += 1
            # print('mu')
            # print(self.mu)
            # print('N')
            # print(self.N)

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
ducb = DUCB(graph, range(10))
ducb.act(1000)
print(ducb.mu)
print(ducb.N)
print('True mus')
print(graph.P.sum(axis=0))