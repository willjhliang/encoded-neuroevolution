import numpy as np


class EAR:
    def __init__(self, ar_N, ln1, ln2, ln3):
        self.ln1, self.ln2, self.ln3 = ln1, ln2, ln3
        self.ar_N = ar_N

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
