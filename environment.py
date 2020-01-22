import numpy as np


# Union-find structure for Connected Components
class CC:
    def __init__(self, n):
        self.n = n
        self.heads = list(range(n))
        self.members = [[i] for i in range(n)]
        self.n_comp = n

    # put the comp of a in the comp of b
    def merge(self, a, b):
        a_comp = self.heads[a]
        b_comp = self.heads[b]
        if a_comp == b_comp:
            return

        self.n_comp -= 1
        for i in self.members[a_comp]:
            self.heads[i] = b_comp
            self.members[b_comp].append(i)
        self.members[a_comp] = []

    def get_size(self, a):
        return len(self.members[self.heads[a]])

    def get_sizes(self):
        return [len(self.members[self.heads[i]]) for i in range(self.n)]


class InfluenceGraph:
    def __init__(self, P):
        assert P.shape[0] == P.shape[1]
        assert (P == P.T).all()
        self.P = P
        self.n = P.shape[0]

    # Returns only the degree, faster
    def observe_degree(self, a):
        r = np.random.rand(self.n-1)
        mask = np.ones(self.n, dtype=bool)
        mask[a] = False
        return sum((r < self.P[a][mask]).astype(int))

    # Return size of CC along with degree, slower
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
        return d, components.get_sizes(), components.n_comp


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


# SBM verifying the assumption 1 of Lugosi et al., k_same is for K_{i,i}, k_diff for K_{i,j}
#  and pops is the lengths of the populations.
class SimpleSBM(SBM):
    def __init__(self, k_diff, k_same, pops):
        self.k_diff = k_diff
        self.k_same = k_same
        S = len(pops)
        K = np.ones((S,S))*k_diff
        np.fill_diagonal(K, k_same)
        print('Alpha optimal for this graph is: ', max(pops)/sum(pops))
        SBM.__init__(self, K, pops)

    def __str__(self):
        return 'k_diff: {}\n' \
               'k_same: {}\n' \
               'n: {}\n' \
               'pops: '.format(self.k_diff, self.k_same, sum(self.pops)) + str(self.pops)


class SimpleCLM(InfluenceGraph):
    def __init__(self, w):
        self.w = w
        InfluenceGraph.__init__(self, np.outer(w, w))

    def __str__(self):
        return 'w: ' + str(self.w)
