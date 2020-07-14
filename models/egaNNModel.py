from scoreModel import ScoreModel
from random import randint
import numpy as np
from collections import Counter


class EGANN:
    def __init__(self, iterations=100, pop_size=100, mut_prob=0.1, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3):
        self.iterations = iterations
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)

        nn = ScoreModel()
        self.ln1, self.ln2, self.ln3 = self.nn.model_dims()
        self.training_data = nn.initial_population()
        self.X = np.array([i[0] for i in self.training_data]).T
        self.y = np.array([i[1] for i in self.training_data]).T

        self.compress_len = self.compress_decoder(self.decoder()).size

        self.clip_lo = -1
        self.clip_hi = 0

    def forward_prop(self, X, model):
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(W2, A1) + b2
        A2 = Z2
        return A2

    def decoder(self):
        deW1 = np.random.randn(4, 3) * 0.01
        deb1 = np.zeros(shape=(4, 1))
        deW2 = np.random.randn(1, 4) * 0.01
        deb2 = np.zeros(shape=(1, 1))
        return {'W1': deW1, 'b1': deb1, 'W2': deW2, 'b2': deb2}

    def compress_decoder(self, decoder):
        deW1 = decoder['W1']
        deb1 = decoder['b1']
        deW2 = decoder['W2']
        deb2 = decoder['b2']

        ret = deW1.flatten()
        ret = np.concatenate((ret, deb1.flatten()))
        ret = np.concatenate((ret, deW2.flatten()))
        ret = np.concatenate((ret, deb2.flatten()))
        return ret

    def expand_decoder(self, decoder):
        deW1, decoder = np.split(decoder, [4 * 3])
        deb1, decoder = np.split(decoder, [4])
        deW2, deb2 = np.split(decoder, [1 * 4])
        deW1 = deW1.reshape(4, 3)
        deb1 = deb1.reshape(4, 1)
        deW2 = deW2.reshape(1, 4)
        deb2 = deb2.reshape(1, 1)
        return {'W1': deW1, 'b1': deb1, 'W2': deW2, 'b2': deb2}

    def decode(self, decoder):
        decoder = self.expand_decoder(decoder)

        L1 = np.zeros((self.ln2, self.ln1 + 1))
        L2 = np.zeros((self.ln3, self.ln2 + 1))
        for i in range(L1.shape[0]):
            for j in range(L1.shape[1]):
                X = np.array([i, j, 0])
                L1[i][j] = self.forward_prop(np.expand_dims(X.T, axis=-1), decoder)
        for i in range(L2.shape[0]):
            for j in range(L2.shape[1]):
                X = np.array([i, j, 1])
                L2[i][j] = self.forward_prop(np.expand_dims(X.T, axis=-1), decoder)
        W1 = L1[:, :-1]
        b1 = np.expand_dims(L1[:, -1], axis=-1)
        W2 = L2[:, :-1]
        b2 = np.expand_dims(L2[:, -1], axis=-1)
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
                J = self.nn.forward_prop(self.X, self.y, self.decode(pop[p]))[0]
                fitness[p] = J
            pop = pop[np.argsort(fitness)]
            fitness = fitness[np.argsort(fitness)]

            J = self.nn.forward_prop(self.X, self.y, self.decode(pop[0]))[0]
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
        model = self.decode(pop[0])
        self.nn.test_model(model)
        np.savez('../saves/egaNNSave.npz', W1=model['W1'], b1=model['b1'], W2=model['W2'], b2=model['b2'])


if __name__ == '__main__':
    egaNN = EGANN()
    egaNN.run()
