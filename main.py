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
        print(self.P[a][mask])
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
        print(components.comps)
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


graph = SimpleSBM(0.05, 0.2, [2,8])
print(graph.observe_full(1))
