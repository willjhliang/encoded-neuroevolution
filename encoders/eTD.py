import numpy as np
import scipy.io


class ETD:
    def __init__(self, td_N, td_sizes, layer_shapes, layer_sizes):
        self.layer_shapes = layer_shapes
        self.layer_sizes = layer_sizes
        self.td_N = td_N  # Same as rank
        self.td_sizes = td_sizes
        # self.mut_scale_V = mut_scale_V
        # self.mut_scale_a = mut_scale_a
        # self.mut_scale_b = mut_scale_b

        # V: valueType_V_layer_rankNum_vectorNum
        # a: valueType_a_layer_rankNum
        self.compressed_id = self.compress_decoder(self.get_decoder(as_id=True))
        self.ids = []
        for i in self.compressed_id:
            if i not in self.ids:
                self.ids.append(i)
        self.ids = np.array(self.ids)
        self.compressed_type = []
        for i in self.compressed_id:
            if int(i[4]) == self.td_N - 1:
                self.compressed_type.append(i[1])
            else:
                self.compressed_type.append('n' + i[1])
        self.compressed_type = np.array(self.compressed_type)

        # self.compressed_id = np.array([])
        # self.compressed_type = np.array([])
        # for layer in range(len(self.td_sizes)):
        #     if self.td_sizes[layer] == []:
        #         continue
        #     for num, typ in enumerate(['W', 'b']):
        #         if np.sum(self.td_sizes[layer][num]) == 0:
        #             continue
        #         size = np.sum(self.td_sizes[layer][num]) * self.td_N
        #         self.compressed_type = np.concatenate((self.compressed_type,
        #                                                np.array(['V'] * size)))
        #         size = self.td_N
        #         self.compressed_type = np.concatenate((self.compressed_type,
        #                                                np.array(['a'] * size)))
        #         # size = self.td_N
        #         # self.compressed_type = np.concatenate((self.compressed_type,
        #         #                                        np.array(['b'] * size)))

        # Used in steprank to protect old values
        # for layer in range(len(self.td_sizes)):
        #     if self.td_sizes[layer] == []:
        #         continue
        #     for num, typ in enumerate(['W', 'b']):
        #         if np.sum(self.td_sizes[layer][num]) == 0:
        #             continue
        #         for vi, v in enumerate(self.td_sizes[layer][num]):
        #             for i in range(self.td_N):
        #                 self.compressed_id = np.concatenate((self.compressed_id,
        #                                                      np.array([typ + 'V' + str(layer) + '_' + str(i) + '_' + str(vi)] * v)))

        #             size = v * self.td_N
        #             self.compressed_type = np.concatenate((self.compressed_type,
        #                                                    np.array(['nV'] * (size - v))))
        #             self.compressed_type = np.concatenate((self.compressed_type,
        #                                                    np.array(['V'] * v)))
        #         for i in range(self.td_N):
        #             self.compressed_id = np.concatenate((self.compressed_id,
        #                                                  np.array([typ + 'a' + str(layer) + '_' + str(i)])))
        #         size = self.td_N
        #         self.compressed_type = np.concatenate((self.compressed_type,
        #                                                np.array(['na'] * (size - 1))))
        #         self.compressed_type = np.concatenate((self.compressed_type,
        #                                                np.array(['a'] * 1)))

        self.size = len(self.compressed_type)

        self.clip_lo = -1
        self.clip_hi = 1

    def get_decoder(self, custom_data=False, as_id=False):
        ret = {}
        for layer in range(len(self.td_sizes)):
            if self.td_sizes[layer] == []:
                continue
            for num, typ in enumerate(['W', 'b']):
                if np.sum(self.td_sizes[layer][num]) == 0:
                    continue
                if not custom_data:
                    # V stores the vectors used in tensor decomp
                    # V[vector number][rank] = 1D vector
                    # Values in V are restricted from clip_lo to clip_hi
                    if not as_id:
                        V = [np.random.uniform(self.clip_lo, self.clip_hi, (self.td_N, v))
                             for v in self.td_sizes[layer][num]]
                    else:
                        V = np.array([])
                        for vi, v in enumerate(self.td_sizes[layer][num]):
                            for i in range(self.td_N):
                                V = np.concatenate((V, np.array([typ + 'V' + str(layer) + '_' + str(i) + '_' + str(vi)] * v)))
                    ret[typ + 'V' + str(layer)] = V

                    # a is the coefficient of output of rank r (M * a)
                    if not as_id:
                        a = np.ones((self.td_N)) / self.td_N
                    else:
                        a = np.array([])
                        for i in range(self.td_N):
                            a = np.concatenate((a, np.array([typ + 'a' + str(layer) + '_' + str(i)])))
                    ret[typ + 'a' + str(layer)] = a

                    # b is the constant of output of rank r (M + b)
                    # ret[typ + 'b' + str(layer)] = np.zeros((self.td_N))
                else:
                    mat = scipy.io.loadmat('cp_decomp_test/factors_16.mat')
                    V = [mat['A'].T, mat['B'].T, mat['C'].T, mat['D'].T]
                    ret[typ + 'V' + str(layer)] = V

                    ret[typ + 'a' + str(layer)] = np.ones((self.td_N))

        return ret

    def compress_decoder(self, decoder):
        ret = np.array([])
        for layer in range(len(self.td_sizes)):
            if self.td_sizes[layer] == []:
                continue
            for num, typ in enumerate(['W', 'b']):
                if np.sum(self.td_sizes[layer][num]) == 0:
                    continue
                V = decoder[typ + 'V' + str(layer)]
                a = decoder[typ + 'a' + str(layer)]
                # b = decoder[typ + 'b' + str(layer)]

                for v in V:
                    ret = np.concatenate((ret, v.flatten()))
                ret = np.concatenate((ret, a))
                # ret = np.concatenate((ret, b))

        return ret

    def expand_decoder(self, decoder):
        ret = {}
        for layer in range(len(self.td_sizes)):
            if self.td_sizes[layer] == []:
                continue
            for num, typ in enumerate(['W', 'b']):
                if np.sum(self.td_sizes[layer][num]) == 0:
                    continue
                V = []
                for v_size in self.td_sizes[layer][num]:
                    v, decoder = np.split(decoder, [self.td_N * v_size])
                    V.append(v.reshape(self.td_N, v_size))
                ret[typ + 'V' + str(layer)] = V
                a, decoder = np.split(decoder, [self.td_N])
                ret[typ + 'a' + str(layer)] = a
                # b, decoder = np.split(decoder, [self.td_N])
                # ret[typ + 'b' + str(layer)] = b

        return ret
        # return ret, decoder[self.size:]

    def decode(self, decoder):
        ret = {}
        for i in range(len(self.layer_shapes)):
            ret['W' + str(i)] = np.zeros(shape=self.layer_shapes[i][0])
            ret['b' + str(i)] = np.zeros(shape=self.layer_shapes[i][1])
        for layer in range(len(self.td_sizes)):
            if self.td_sizes[layer] == []:
                continue
            for r in range(self.td_N):
                for num, typ in enumerate(['W', 'b']):
                    if np.sum(self.td_sizes[layer][num]) == 0:
                        continue
                    V = decoder[typ + 'V' + str(layer)]
                    a = decoder[typ + 'a' + str(layer)][r]
                    # b = decoder[typ + 'b' + str(layer)][r]
                    layer_shape = self.layer_shapes[layer][num]
                    layer_size = self.layer_sizes[layer][num]

                    M = V[0][r]
                    for v in V[1:]:
                        M = M[..., None] * v[r].T

                    M = M * a
                    # M = M * a + b
                    M, _ = np.split(M.flatten(), [layer_size])
                    ret[typ + str(layer)] += M.reshape(layer_shape)

        return ret

    def clip(self, x):
        # x, y = np.split(x, [self.size])  # y is unrelated to current encoder

        z = x[self.compressed_type == 'V']
        z[z < self.clip_lo] = self.clip_lo
        z[z > self.clip_hi] = self.clip_hi
        x[self.compressed_type == 'V'] = z

        return x
        # return x, y

    # def cross(self, x, y, prob):
    #     c1 = x.copy()
    #     c2 = y.copy()
    #     swap = np.random.rand(*self.ids.shape) < prob
    #     for i in np.where(swap)[0]:
    #         curr = self.ids[i]
    #         match = self.compressed_id == curr
    #         if self.compressed_type[i] == 'V' or self.compressed_type[i] == 'a':
    #             c1[match] = y[match].copy()
    #             c2[match] = x[match].copy()

    #     c1 = self.clip(c1)
    #     c2 = self.clip(c2)

    #     return c1, c2

    # def mut(self, x, prob):
    #     # x, y = np.split(x, [self.size])  # y is unrelated to current encoder
    #     rand = np.random.rand(self.size)
    #     mut = x[rand < prob]
    #     mut_type = self.compressed_type[rand < prob]
    #     mut[mut_type == 'V'] += np.random.normal(0, self.mut_scale_V,
    #                                              np.shape(mut[mut_type == 'V']))
    #     mut[mut_type == 'a'] += np.random.normal(0, self.mut_scale_a,
    #                                              np.shape(mut[mut_type == 'a']))
    #     x[rand < prob] = mut

    #     return x
        # return x, y
