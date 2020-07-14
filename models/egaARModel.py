from scoreModel import ScoreModel
from actionModel import ActionModel
import numpy as np


class EGAAR:
    def __init__(self, iterations=100, pop_size=100, mut_prob=0.7, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, ar_N=2):
        self.iterations = iterations
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)

        self.nn = ActionModel()
        self.ln1, self.ln2, self.ln3 = self.nn.model_dims()
        self.training_data = self.nn.training_data
        self.X = np.array([i[0] for i in self.training_data]).T
        self.y = np.array([i[1] for i in self.training_data]).T

        self.ar_N = ar_N
        self.compress_len = self.compress_decoder(self.decoder()).size

        self.clip_lo = -1.1
        self.clip_hi = 1.1

    def decoder(self):
        ret = {}
        for i in range(self.ar_N):
            ret['a1' + str(i)] = np.random.rand(1) * 2 - 1
            ret['c1' + str(i)] = np.random.rand(1) * 2 - 1
            ret['a2' + str(i)] = np.random.rand(1) * 2 - 1
            ret['c2' + str(i)] = np.random.rand(1) * 2 - 1
        return ret

    def compress_decoder(self, decoder):
        ret = np.array([])
        for i in range(self.ar_N):
            a1 = decoder['a1' + str(i)]
            c1 = decoder['c1' + str(i)]
            a2 = decoder['a2' + str(i)]
            c2 = decoder['c2' + str(i)]
            ret = np.concatenate((ret, a1))
            ret = np.concatenate((ret, c1))
            ret = np.concatenate((ret, a2))
            ret = np.concatenate((ret, c2))
        return ret

    def expand_decoder(self, decoder):
        ret = {}
        for i in range(self.ar_N):
            a1, decoder = np.split(decoder, [1])
            c1, decoder = np.split(decoder, [1])
            a2, decoder = np.split(decoder, [1])
            c2, decoder = np.split(decoder, [1])
            ret['a1' + str(i)] = a1.reshape(1)
            ret['c1' + str(i)] = c1.reshape(1)
            ret['a2' + str(i)] = a2.reshape(1)
            ret['c2' + str(i)] = c2.reshape(1)
        return ret

    def decode(self, decoder):
        decoder = self.expand_decoder(decoder)

        W1 = np.zeros(shape=(self.ln2, self.ln1))
        b1 = np.zeros(shape=(self.ln2, 1))
        W2 = np.zeros(shape=(self.ln3, self.ln2))
        b2 = np.zeros(shape=(self.ln3, 1))

        for i in range(self.ar_N):
            a1 = decoder['a1' + str(i)]
            c1 = decoder['c1' + str(i)]
            a2 = decoder['a2' + str(i)]
            c2 = decoder['c2' + str(i)]

            L1 = np.zeros(shape=(self.ln2, self.ln1 + 1)).flatten()
            L2 = np.zeros(shape=(self.ln3, self.ln2 + 1)).flatten()
            L1[0] = c1
            for i in range(1, L1.shape[0]):
                L1[i] = L1[i - 1] * a1
            L2[0] = c2
            for i in range(1, L2.shape[0]):
                L2[i] = L2[i - 1] * a2

            L1 = L1.reshape(self.ln2, self.ln1 + 1)
            L2 = L2.reshape((self.ln3, self.ln2 + 1))

            W1 += L1[:, :-1]
            b1 += np.expand_dims(L1[:, -1], axis=-1)
            W2 += L2[:, :-1]
            b2 += np.expand_dims(L2[:, -1], axis=-1)
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
            pop[p] = self.compress_decoder(self.decoder())
            pop[p] = self.clip(pop[p])

        for t in range(1, self.iterations + 1):
            fitness = np.zeros(self.pop_size)
            for p in range(self.pop_size):
                J = self.nn.forward_prop(self.decode(pop[p]), self.X, self.y)[0]
                fitness[p] = J
            pop = pop[np.argsort(fitness)]
            fitness = fitness[np.argsort(fitness)]

            J = self.nn.forward_prop(self.decode(pop[0]), self.X, self.y)[0]
            if t % 25 == 0:
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
        model = self.decode(pop[0])
        self.nn.test(model)
        np.savez('../saves/egaARSave.npz', W1=model['W1'], b1=model['b1'], W2=model['W2'], b2=model['b2'])


if __name__ == '__main__':
    GA = EGAAR()
    GA.run()
