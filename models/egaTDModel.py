from scoreModel import ScoreModel
from actionModel import ActionModel
from simulationModel import SimulationModel

from random import randint
import numpy as np
from collections import Counter


class EGATD:
    def __init__(self, iterations=500, pop_size=100, mut_prob=0.3, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, td_N=2):
        self.iterations = iterations
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)

        self.problem = SimulationModel()
        self.ln1, self.ln2, self.ln3 = self.problem.model_dims()
        # self.training_data = self.problem.training_data
        # self.X = np.array([i[0] for i in self.training_data]).T
        # self.y = np.array([i[1] for i in self.training_data]).T

        self.td_N = td_N
        self.td_size1 = 3
        self.td_size2 = 5
        self.td_size3 = 11
        self.compress_len = self.compress_decoder(self.decoder()).size

        self.clip_lo = -1
        self.clip_hi = 0

    def decoder(self):
        ret = {}
        for i in range(self.td_N):
            ret['V1' + str(i)] = np.random.randn(self.td_size1, 1) * 0.01
            ret['V2' + str(i)] = np.random.randn(self.td_size2, 1) * 0.01
            ret['V3' + str(i)] = np.random.randn(self.td_size3, 1) * 0.01
            ret['C' + str(i)] = np.array([1]).reshape(1, 1)
        return ret

    def compress_decoder(self, decoder):
        ret = np.array([])
        for i in range(self.td_N):
            V1 = decoder['V1' + str(i)][:, 0]
            V2 = decoder['V2' + str(i)][:, 0]
            V3 = decoder['V3' + str(i)][:, 0]
            C = decoder['C' + str(i)][:, 0]
            ret = np.concatenate((ret, V1))
            ret = np.concatenate((ret, V2))
            ret = np.concatenate((ret, V3))
            ret = np.concatenate((ret, C))
        return ret

    def expand_decoder(self, decoder):
        ret = {}
        for i in range(self.td_N):
            V1, decoder = np.split(decoder, [self.td_size1 * 1])
            V2, decoder = np.split(decoder, [self.td_size2 * 1])
            V3, decoder = np.split(decoder, [self.td_size3 * 1])
            C, decoder = np.split(decoder, [1 * 1])
            ret['V1' + str(i)] = V1.reshape(self.td_size1, 1)
            ret['V2' + str(i)] = V2.reshape(self.td_size2, 1)
            ret['V3' + str(i)] = V3.reshape(self.td_size3, 1)
            ret['C' + str(i)] = C.reshape(1, 1)
        return ret

    def decode(self, decoder):
        decoder = self.expand_decoder(decoder)

        W1 = np.zeros(shape=(self.ln2, self.ln1))
        b1 = np.zeros(shape=(self.ln2, 1))
        W2 = np.zeros(shape=(self.ln3, self.ln2))
        b2 = np.zeros(shape=(self.ln3, 1))

        for i in range(self.td_N):
            V1 = decoder['V1' + str(i)]
            V2 = decoder['V2' + str(i)]
            V3 = decoder['V3' + str(i)]
            C = decoder['C' + str(i)]

            T = np.dot(V1, V2.T)
            T = T[..., None] * V3[:, 0]
            T = T * C
            T = T.flatten()

            L1, T = np.split(T, [self.ln2 * (self.ln1 + 1)])
            L2, T = np.split(T, [self.ln3 * (self.ln2 + 1)])
            L1 = L1.reshape(self.ln2, self.ln1 + 1)
            L2 = L2.reshape((self.ln3, self.ln2 + 1))

            W1 += L1[:, :-1]
            b1 += np.expand_dims(L1[:, -1], axis=-1)
            W2 += L2[:, :-1]
            b2 += np.expand_dims(L2[:, -1], axis=-1)
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def clip(self, x):
        # x[x < self.clip_lo] = self.clip_lo
        # x[x > self.clip_hi] = self.clip_hi
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
            pop[p] = self.compress_decoder(self.decoder())
            pop[p] = self.clip(pop[p])

        for t in range(1, self.iterations + 1):
            fitness = np.zeros(self.pop_size)
            for p in range(self.pop_size):
                J = self.problem.test(self.decode(pop[p]), 3)[0]
                fitness[p] = J
            k = -1 if isinstance(self.problem, SimulationModel) else 1  # Maximize metric in SimulationModel, minimize error otherwise
            pop = pop[np.argsort(k * fitness)]
            fitness = fitness[np.argsort(k * fitness)]

            J, steps, score = self.problem.test(self.decode(pop[0]), 3)
            if t % 25 == 0:
                print('Iter ' + str(t).zfill(3) + ': ' + str(J) + ' ' + str(steps) + ' ' + str(score))

            prob = np.array(fitness, copy=True) - min(0, np.amin(fitness))  # prevent negatives
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
        model = self.decode(pop[0])
        self.problem.test(model)
        np.savez('../saves/egaTDSave.npz', W1=model['W1'], b1=model['b1'], W2=model['W2'], b2=model['b2'])


if __name__ == '__main__':
    GA = EGATD()
    GA.run()
