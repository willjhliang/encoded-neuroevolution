
from snakeNN import snakeNN
import numpy as np


class GeneticAlgo:
    def __init__(self, iterations=100, pop_size=100, mut_prob=0.1, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3):
        self.iterations = 100
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)
        self.compress_len = 25 * 5 + 25 + 25 * 1 + 1

    def compress(W1, b1, W2, b2):
        ret = W1.flatten()
        ret = np.concatenate(ret, b1.flatten())
        ret = np.concatenate(ret, W2.flatten())
        ret = np.concatenate(ret, b2.flatten())
        return ret

    def expand(arr):
        W1, arr = np.split(arr, 25 * 5)
        b1, arr = np.split(arr, 25)
        W2, b2 = np.split(arr, 25 * 1)
        W1 = W1.reshape(25, 5)
        b1 = b1.reshape(25, 1)
        W2 = W2.reshape(1, 25)
        b2 = b2.reshape(1, 1)
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def cross(self, x, y):
        c1 = x.copy()
        c2 = y.copy()
        for i in range(self.compress_len):
            if np.random.random() < 0.5:
                c1[i] = y[i].copy()
                c2[i] = x[i].copy()
        return c1, c2

    def mut(self, c):
        for i in range(self.compress_len):
            if np.random.random() < self.mut_prob:
                c[i] += np.random.uniform(low=-5.0, high=5.0)
        return c

    def run(self):
        pop = np.array([np.zeros(self.compress_len)] * self.pop_size)
        for p in range(self.pop_size):
            W1 = np.random.randn(25, 5) * 0.01
            b1 = np.zeros(shape=(25, 1))
            W2 = np.random.randn(1, 25) * 0.01
            b2 = np.zeros(shape=(1, 1))
            pop[p] = self.compress(W1, b1, W2, b2)

        t = 1
        while t <= self.iterations:
            fitness = np.zeros(self.pop_size)
            for p in range(self.pop_size):
                fitness[p] = snakeNN.test_model(self.expand(pop[p]))
            pop = pop[np.argsort(fitness)]
            fitness = fitness[np.argsort(fitness)]

            prob = np.array(fitness, copy=True)
            prob = prob / np.sum(prob)
            cum_prob = np.cumsum(prob)
            par = np.array([np.zeros(self.compress_len)] * self.par_size)
            for i in range(self.elite_size):
                par[i] = pop[i].copy()
            for i in range(self.elite_size, self.par_size):
                idx = np.searchsorted(cum_prob, np.random.random())
                par[i] = pop[idx].copy()

            eff = np.array([False] * self.par_size)
            par_ct = 0
            while par_ct < 1:
                for i in range(self.par_size):
                    if np.random.random() < self.cross_prob:
                        eff[i] = True
                        par_ct += 1
            eff_par = par[eff].copy()

            pop = np.array([np.zeros(self.compress_len)] * self.pop_size)
            for i in range(self.par_size):
                pop[i] = par[i].copy()
            for i in range(self.par_size, self.pop_size, 2):
                p1 = eff_par[np.random.randint(0, par_ct):].copy()
                p2 = eff_par[np.random.randint(0, par_ct):].copy()
                pop[i], pop[i + 1] = self.cross(p1, p2)
                pop[i] = self.mut(pop[i])
                pop[i + 1] = self.mut(pop[i + 1])
