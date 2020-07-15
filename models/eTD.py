import numpy as np


class ETD:
    def __init__(self, td_N, td_size1, td_size2, td_size3, ln1, ln2, ln3):
        self.ln1, self.ln2, self.ln3 = ln1, ln2, ln3
        self.td_N = td_N
        self.td_size1 = 3
        self.td_size2 = 5
        self.td_size3 = 11

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
