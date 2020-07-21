import numpy as np


class ETD:
    def __init__(self, td_N, td_size1, td_size2, td_size3, ln):
        self.ln = ln
        self.td_N = td_N
        self.td_size1 = td_size1
        self.td_size2 = td_size2
        self.td_size3 = td_size3

        self.clip_lo = -1
        self.clip_hi = 1

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
        ret = {}
        for j in range(1, len(self.ln)):
            ret['W' + str(j)] = np.zeros(shape=(self.ln[j], self.ln[j - 1]))
            ret['b' + str(j)] = np.zeros(shape=(self.ln[j], 1))
        for i in range(self.td_N):
            V1 = decoder['V1' + str(i)]
            V2 = decoder['V2' + str(i)]
            V3 = decoder['V3' + str(i)]
            C = decoder['C' + str(i)]
            T = np.dot(V1, V2.T)
            T = T[..., None] * V3[:, 0]
            T = T * C
            T = T.flatten()

            for j in range(1, len(self.ln)):
                L, T = np.split(T, [self.ln[j] * (self.ln[j - 1] + 1)])
                L = L.reshape(self.ln[j], self.ln[j - 1] + 1)
                ret['W' + str(j)] += L[:, :-1]
                ret['b' + str(j)] += np.expand_dims(L[:, -1], axis=-1)

        return ret

    def clip(self, x):
        # x[x < self.clip_lo] = self.clip_lo
        # x[x > self.clip_hi] = self.clip_hi

        return x
