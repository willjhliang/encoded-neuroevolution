import numpy as np


class ERand:
    def __init__(self, rand_N, ln):
        self.ln = ln
        self.rand_N = rand_N

        self.clip_lo = -1.1
        self.clip_hi = 1.1

    def decoder(self):
        ret = {}
        for i in range(self.rand_N):
            for j in range(1, len(self.ln)):
                ret['Rand' + str(j) + str(i)] = (np.random.rand(1) * 100).round()

        return ret

    def compress_decoder(self, decoder):
        ret = np.array([])
        for i in range(self.rand_N):
            for j in range(1, len(self.ln)):
                r = decoder['Rand' + str(j) + str(i)]
                ret = np.concatenate((ret, r))

        return ret

    def expand_decoder(self, decoder):
        ret = {}
        for i in range(self.rand_N):
            for j in range(1, len(self.ln)):
                r, decoder = np.split(decoder, [1])
                ret['Rand' + str(j) + str(i)] = r.reshape(1)

        return ret

    def decode(self, decoder):
        ret = {}
        for j in range(1, len(self.ln)):
            ret['W' + str(j)] = np.zeros(shape=(self.ln[j], self.ln[j - 1]))
            ret['b' + str(j)] = np.zeros(shape=(self.ln[j], 1))
        for i in range(self.rand_N):
            for j in range(1, len(self.ln)):
                r = decoder['Rand' + str(j) + str(i)]
                L = np.zeros(shape=(self.ln[j], self.ln[j] + 1)).flatten()
                np.random.seed(int(r[0]))
                for k in range(L.shape[0]):
                    r = np.random.random()
                    L[k] = r * 2 - 1
                    np.random.seed(int(r * 100))

                L = L.reshape(self.ln[j], self.ln[j] + 1)
                ret['W' + str(j)] += L[:, :-1]
                ret['b' + str(j)] = np.expand_dims(L[:, -1], axis=-1)

        return ret

    def clip(self, x):
        # x[x < self.clip_lo] = self.clip_lo
        # x[x > self.clip_hi] = self.clip_hi

        return x
