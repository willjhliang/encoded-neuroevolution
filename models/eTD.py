import numpy as np


class ETD:
    def __init__(self, td_N, td_sizes, layer_shapes, layer_sizes):
        self.layer_shapes = layer_shapes
        self.layer_sizes = layer_sizes
        self.td_N = td_N
        self.td_sizes = td_sizes
        self.size = len(self.compress_decoder(self.decoder()))

        self.noclip = np.array([])
        for j in range(td_N):
            for i in td_sizes:
                if i == []:
                    continue
                self.noclip = np.concatenate((self.noclip, np.zeros(np.sum(i))))
                self.noclip = np.concatenate((self.noclip, np.ones(4)))
        self.clip_lo = 0
        self.clip_hi = 1

    def decoder(self):
        ret = {}
        for i in range(self.td_N):
            for j in range(len(self.td_sizes)):
                if self.td_sizes[j] == []:
                    continue
                for num, typ in enumerate(['W', 'b']):
                    for k in range(1, 4):
                        ret[typ + 'V' + str(k) + str(j) + str(i)] = np.random.randn(self.td_sizes[j][num][k - 1], 1)
                    ret[typ + 'a' + str(j) + str(i)] = np.zeros((1, 1))
                    ret[typ + 'c' + str(j) + str(i)] = np.array([0.1 / self.td_N]).reshape(1, 1)

        return ret

    def compress_decoder(self, decoder):
        ret = np.array([])
        for i in range(self.td_N):
            for j in range(len(self.td_sizes)):
                if self.td_sizes[j] == []:
                    continue
                for typ in ['W', 'b']:
                    for k in range(1, 4):
                        V = decoder[typ + 'V' + str(k) + str(j) + str(i)][:, 0]
                        ret = np.concatenate((ret, V))
                    a = decoder[typ + 'a' + str(j) + str(i)][:, 0]
                    c = decoder[typ + 'c' + str(j) + str(i)][:, 0]
                    ret = np.concatenate((ret, a))
                    ret = np.concatenate((ret, c))

        return ret

    def expand_decoder(self, decoder):
        ret = {}
        for i in range(self.td_N):
            for j in range(len(self.td_sizes)):
                if self.td_sizes[j] == []:
                    continue
                for num, typ in enumerate(['W', 'b']):
                    for k in range(1, 4):
                        V, decoder = np.split(decoder,
                                              [self.td_sizes[j][num][k - 1]])
                        ret[typ + 'V' + str(k) + str(j) + str(i)] = V.reshape(self.td_sizes[j][num][k - 1], 1)
                    a, decoder = np.split(decoder, [1])
                    ret[typ + 'a' + str(j) + str(i)] = a.reshape(1, 1)
                    c, decoder = np.split(decoder, [1])
                    ret[typ + 'c' + str(j) + str(i)] = c.reshape(1, 1)

        return ret, decoder

    def decode(self, decoder):
        ret = {}
        for i in range(1, len(self.layer_shapes)):
            ret['W' + str(i)] = np.zeros(shape=self.layer_shapes[i][0])
            ret['b' + str(i)] = np.zeros(shape=self.layer_shapes[i][1])
        for i in range(self.td_N):
            for j in range(len(self.td_sizes)):
                if self.td_sizes[j] == []:
                    continue
                for num, typ in enumerate(['W', 'b']):
                    V1 = decoder[typ + 'V1' + str(j) + str(i)]
                    V2 = decoder[typ + 'V2' + str(j) + str(i)]
                    V3 = decoder[typ + 'V3' + str(j) + str(i)]
                    a = decoder[typ + 'a' + str(j) + str(i)]
                    c = decoder[typ + 'c' + str(j) + str(i)]

                    T = np.dot(V1, V2.T)
                    T = T[..., None] * V3[:, 0]
                    T = T * a + c
                    T = T.flatten()

                    L, T = np.split(T, [np.prod(self.layer_sizes[j][num])])
                    ret[typ + str(j)] += L.reshape(self.layer_shapes[j][num])

        return ret

    def clip(self, x):
        x, y = np.split(x, [self.size])
        z = x[self.noclip == 0]
        z[z < self.clip_lo] = self.clip_lo
        z[z > self.clip_hi] = self.clip_hi
        x[self.noclip == 0] = z

        return x, y

    def mut(self, x, prob):
        x, y = np.split(x, [self.size])
        rand = np.random.rand(self.size)
        mut = x[rand < prob]
        mut_noclip = self.noclip[rand < prob]
        mut[mut_noclip == 0] = np.random.uniform()
        mut[mut_noclip == 1] += np.random.normal(scale=10)
        x[rand < prob] = mut

        return x, y
