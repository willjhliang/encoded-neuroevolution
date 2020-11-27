import numpy as np


class EAR:
    def __init__(self, ar_N, ln):
        self.ln = ln
        self.ar_N = ar_N

        self.clip_lo = -1.1
        self.clip_hi = 1.1

    def decoder(self):
        ret = {}
        for i in range(self.ar_N):
            for j in range(1, len(self.ln)):
                ret['a' + str(j) + str(i)] = np.random.rand(1) * 2 - 1
                ret['c' + str(j) + str(i)] = np.random.rand(1) * 2 - 1
                ret['C' + str(j) + str(i)] = np.array([1]).reshape(1, 1)

        return ret

    def compress_decoder(self, decoder):
        ret = np.array([])
        for i in range(self.ar_N):
            for j in range(1, len(self.ln)):
                a = decoder['a' + str(j) + str(i)]
                c = decoder['c' + str(j) + str(i)]
                C = decoder['C' + str(j) + str(i)]
                ret = np.concatenate((ret, a))
                ret = np.concatenate((ret, c))
                ret = np.concatenate((ret, C))

        return ret

    def expand_decoder(self, decoder):
        ret = {}
        for i in range(self.ar_N):
            for j in range(1, len(self.ln)):
                a, decoder = np.split(decoder, [1])
                c, decoder = np.split(decoder, [1])
                C, deocder = np.split(decoder, [1])
                ret['a' + str(j) + str(i)] = a.reshape(1)
                ret['c' + str(j) + str(i)] = c.reshape(1)
                ret['C' + str(j) + str(i)] = c.reshape(1)

        return ret

    def decode(self, decoder):
        ret = {}
        for j in range(1, len(self.ln)):
            ret['W' + str(j)] = np.zeros(shape=(self.ln[j], self.ln[j - 1]))
            ret['b' + str(j)] = np.zeros(shape=(self.ln[j], 1))
        for i in range(self.ar_N):
            for j in range(1, len(self.ln)):
                a = decoder['a' + str(j) + str(i)]
                c = decoder['c' + str(j) + str(i)]
                C = decoder['C' + str(j) + str(i)]

                L = np.zeros(shape=(self.ln[j], self.ln[j - 1] + 1)).flatten()
                L[0] = c
                for k in range(1, L.shape[0]):
                    L[k] = L[k - 1] * a
                L *= C

                L = L.reshape(self.ln[j], self.ln[j - 1] + 1)
                ret['W' + str(j)] += L[:, :-1]
                ret['b' + str(j)] += np.expand_dims(L[:, -1], axis=-1)

        return ret

    def clip(self, x):
        x[x < self.clip_lo] = self.clip_lo
        x[x > self.clip_hi] = self.clip_hi

        return x
