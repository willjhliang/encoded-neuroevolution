from scoreModel import ScoreModel
from random import randint
import numpy as np
from collections import Counter


class GA:
    def __init__(self, iterations=100, pop_size=100, mut_prob=0.2, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3):
        self.iterations = iterations
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)

        self.nn = ScoreModel()
        self.training_data = self.nn.initial_population()
        self.X = np.array([i[0] for i in self.training_data]).T
        self.y = np.array([i[1] for i in self.training_data]).T
        self.compress_len = self.compress(self.nn.model()).size

        self.clip_lo = -1
        self.clip_hi = 0

    def compress(self, model):
        # W1 = model['W1']
        # b1 = model['b1']

        # ret = W1.flatten()
        # ret = np.concatenate((ret, b1.flatten()))
        # return ret

        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']

        ret = W1.flatten()
        ret = np.concatenate((ret, b1.flatten()))
        ret = np.concatenate((ret, W2.flatten()))
        ret = np.concatenate((ret, b2.flatten()))
        return ret

    def expand(self, arr):
        # W1, b1 = np.split(arr, [1 * 4])
        # W1 = W1.reshape(1, 4)
        # b1 = b1.reshape(1, 1)
        # return {'W1': W1, 'b1': b1}

        W1, arr = np.split(arr, [25 * 5])
        b1, arr = np.split(arr, [25])
        W2, b2 = np.split(arr, [25 * 1])
        W1 = W1.reshape(25, 5)
        b1 = b1.reshape(25, 1)
        W2 = W2.reshape(1, 25)
        b2 = b2.reshape(1, 1)
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def clip(self, x):
        x[x < self.clip_lo] = self.clip_lo
        x[x > self.clip_hi] = self.clip_hi
        return x

    def cross(self, x, y):
        c1 = x.copy()
        c2 = y.copy()
        for i in range(self.compress_len):
            if np.random.random() < 0.5:
                c1[i] = y[i].copy()
                c2[i] = x[i].copy()
        c1 = self.clip(c1)
        c2 = self.clip(c2)
        return c1, c2

    def mut(self, c):
        for i in range(self.compress_len):
            if np.random.random() < self.mut_prob:
                c[i] += np.random.normal(scale=0.1)
                # c[i] = np.random.uniform(low=-0.05, high=0.8)
        c = self.clip(c)
        return c

    def run(self):
        pop = np.array([np.zeros(self.compress_len)] * self.pop_size)
        for p in range(self.pop_size):
            pop[p] = self.compress(self.nn.model())
            pop[p] = self.clip(pop[p])

        for t in range(1, self.iterations + 1):
            fitness = np.zeros(self.pop_size)
            for p in range(self.pop_size):
                J = self.nn.forward_prop(self.X, self.y, self.expand(pop[p]))[0]
                fitness[p] = J
            pop = pop[np.argsort(fitness)]
            fitness = fitness[np.argsort(fitness)]

            J = self.nn.forward_prop(self.X, self.y, self.expand(pop[0]))[0]
            if t % 10 == 0:
                print('Iter ' + str(t).zfill(3) + ': ' + str(J))

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
                p1 = eff_par[np.random.randint(0, par_ct)].copy()
                p2 = eff_par[np.random.randint(0, par_ct)].copy()
                pop[i], pop[i + 1] = self.cross(p1, p2)
                pop[i] = self.mut(pop[i])
                pop[i + 1] = self.mut(pop[i + 1])
        model = self.expand(pop[0])
        self.nn.test_model(model)
        np.savez('../saves/gaSave.npz', W1=model['W1'], b1=model['b1'], W2=model['W2'], b2=model['b2'])


if __name__ == '__main__':
    GA = GA()
    GA.run()
