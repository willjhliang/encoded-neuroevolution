import numpy as np


class ENN:
    def __init__(self, nn_N, eln1, eln2, eln3, ln):
        self.ln = ln
        self.nn_N = nn_N
        self.eln1, self.eln2, self.eln3 = eln1, eln2, eln3

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
        ret = {}
        for i in range(self.nn_N):
            ret['W1' + str(i)] = np.random.randn(self.eln2, self.eln1) * 0.01
            ret['b1' + str(i)] = np.zeros(shape=(self.eln2, 1))
            ret['W2' + str(i)] = np.random.randn(self.eln3, self.eln2) * 0.01
            ret['b2' + str(i)] = np.zeros(shape=(self.eln3, 1))
        return ret

    def compress_decoder(self, decoder):
        ret = np.array([])
        for i in range(self.nn_N):
            deW1 = decoder['W1' + str(i)]
            deb1 = decoder['b1' + str(i)]
            deW2 = decoder['W1' + str(i)]
            deb2 = decoder['b1' + str(i)]
            ret = np.concatenate((ret, deW1.flatten()))
            ret = np.concatenate((ret, deb1.flatten()))
            ret = np.concatenate((ret, deW2.flatten()))
            ret = np.concatenate((ret, deb2.flatten()))

        return ret

    def expand_decoder(self, decoder):
        ret = {}
        for i in range(self.nn_N):
            deW1, decoder = np.split(decoder, [self.eln2 * self.eln1])
            deb1, decoder = np.split(decoder, [self.eln2 * 1])
            deW2, decoder = np.split(decoder, [self.eln3 * self.eln2])
            deb2, decoder = np.split(decoder, [self.eln3 * 1])
            ret['W1' + str(i)] = deW1.reshape(self.eln2, self.eln1)
            ret['b1' + str(i)] = deb1.reshape(self.eln2, 1)
            ret['W2' + str(i)] = deW2.reshape(self.eln3, self.eln2)
            ret['b2' + str(i)] = deb2.reshape(self.eln3, 1)

        return ret

    def decode(self, decoder):
        ret = {}
        for j in range(1, len(self.ln)):
            ret['W' + str(j)] = np.zeros(shape=(self.ln[j], self.ln[j - 1]))
            ret['b' + str(j)] = np.zeros(shape=(self.ln[j], 1))
        for i in range(self.nn_N):
            for j in range(1, len(self.ln)):
                L = np.zeros((self.ln[j], self.ln[j - 1] + 1))
                for k in range(L.shape[0]):
                    for l in range(L.shape[1]):
                        X = np.array([k, l, j])
                        L[k][l] += self.forward_prop(np.expand_dims(X.T, axis=-1), decoder)
                ret['W' + str(j)] += L[:, :-1]
                ret['b' + str(j)] += np.expand_dims(L[:, -1], axis=-1)

        return ret

    def clip(self, x):
        # x[x < self.clip_lo] = self.clip_lo
        # x[x > self.clip_hi] = self.clip_hi
        return x
