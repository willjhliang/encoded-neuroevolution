import numpy as np


class ERand:
    def __init__(self, rand_N, ln1, ln2, ln3):
        self.ln1, self.ln2, self.ln3 = ln1, ln2, ln3
        self.rand_N = rand_N

        self.clip_lo = -1.1
        self.clip_hi = 1.1

    def decoder(self):
        ret = {}
        for i in range(self.rand_N):
            ret['Rand1' + str(i)] = (np.random.rand(1) * 100).round()
            ret['Rand2' + str(i)] = (np.random.rand(1) * 100).round()

        return ret

    def compress_decoder(self, decoder):
        ret = np.array([])
        for i in range(self.rand_N):
            r1 = decoder['Rand1' + str(i)]
            r2 = decoder['Rand2' + str(i)]
            ret = np.concatenate((ret, r1))
            ret = np.concatenate((ret, r2))

        return ret

    def expand_decoder(self, decoder):
        ret = {}
        for i in range(self.rand_N):
            r1, decoder = np.split(decoder, [1])
            r2, decoder = np.split(decoder, [1])
            ret['Rand1' + str(i)] = r1.reshape(1)
            ret['Rand2' + str(i)] = r2.reshape(1)

        return ret

    def decode(self, decoder):
        W1 = np.zeros(shape=(self.ln2, self.ln1))
        b1 = np.zeros(shape=(self.ln2, 1))
        W2 = np.zeros(shape=(self.ln3, self.ln2))
        b2 = np.zeros(shape=(self.ln3, 1))

        for i in range(self.rand_N):
            r1 = decoder['Rand1' + str(i)]
            r2 = decoder['Rand2' + str(i)]

            L1 = np.zeros(shape=(self.ln2, self.ln1 + 1)).flatten()
            L2 = np.zeros(shape=(self.ln3, self.ln2 + 1)).flatten()
            np.random.seed(int(r1[0]))
            for i in range(L1.shape[0]):
                k = np.random.random()
                L1[i] = k * 2 - 1
                np.random.seed(int(k * 100))
            np.random.seed(int(r2[0]))
            for i in range(L2.shape[0]):
                k = np.random.random()
                L2[i] = k * 2 - 1
                np.random.seed(int(k * 100))

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
