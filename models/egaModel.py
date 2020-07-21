
import sys
sys.path.insert(0, '../problems')

from scoreProblem import ScoreProb
from actionProblem import ActionProb

from eAR import EAR
from eTD import ETD
from eRand import ERand
from eNN import ENN

import numpy as np
import tensorflow as tf


class EGA:
    def __init__(self, problem, ar_N, td_N, rand_N, nn_N, p, c, iterations=100, pop_size=100,
                 mut_prob=0.3, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, file_name='default'):
        self.iterations = iterations
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)
        self.file_name = file_name

        self.problem = None
        if problem == 'Score':
            self.problem = ScoreProb(p, c)
        if problem == 'Action':
            self.problem = ActionProb(p, c)
        self.ln = self.problem.model_dims()
        self.lfunc = ['ReLU', '']

        self.count = 0

        self.td_N = td_N
        self.ar_N = ar_N
        self.rand_N = rand_N
        self.nn_N = nn_N

        # For 400 - 40 - 3
        self.td_size1 = 19
        self.td_size2 = 23
        self.td_size3 = 37

        # For 160 - 40 - 20 - 3
        # self.td_size1 = 11
        # self.td_size2 = 18
        # self.td_size3 = 37

        # For 160 - 40 - 3
        # self.td_size1 = 13
        # self.td_size2 = 22
        # self.td_size3 = 23

        # For 160 - 20 - 3
        # self.td_size1 = 11
        # self.td_size2 = 13
        # self.td_size3 = 23

        # For 120 - 20 - 3
        # self.td_size1 = 6
        # self.td_size2 = 18
        # self.td_size3 = 23

        # For 4 - 20 - 3
        # self.td_size1 = 3
        # self.td_size2 = 5
        # self.td_size3 = 11

        self.eln1 = 3
        self.eln2 = 4
        self.eln3 = 1

        self.ar_size = 4 * self.ar_N
        self.td_size = (self.td_size1 + self.td_size2 + self.td_size3 + 1) * self.td_N
        self.rand_size = 2 * self.rand_N
        self.nn_size = (self.eln2 * self.eln1 + self.eln2 + self.eln3 * self.eln2 + self.eln3) * self.nn_N

        self.ar = EAR(self.ar_N, self.ln)
        self.td = ETD(self.td_N, self.td_size1, self.td_size2, self.td_size3, self.ln)
        self.rand = ERand(self.rand_N, self.ln)
        self.nn = ENN(self.nn_N, self.eln1, self.eln2, self.eln3, self.ln)
        self.compress_len = self.compress_decoder(self.decoder()).size

        self.clip_lo = -1.1
        self.clip_hi = 1.1

    def decoder(self):
        ret = {'id': self.count}
        self.count += 1
        tret = self.ar.decoder()
        for key in tret:
            ret[key] = tret[key]
        tret = self.td.decoder()
        for key in tret:
            ret[key] = tret[key]
        tret = self.rand.decoder()
        for key in tret:
            ret[key] = tret[key]
        tret = self.nn.decoder()
        for key in tret:
            ret[key] = tret[key]

        return ret

    def compress_decoder(self, decoder):
        ret = np.array([decoder['id']])
        tret = self.ar.compress_decoder(decoder)
        ret = np.concatenate((ret, tret))
        tret = self.td.compress_decoder(decoder)
        ret = np.concatenate((ret, tret))
        tret = self.rand.compress_decoder(decoder)
        ret = np.concatenate((ret, tret))
        tret = self.nn.compress_decoder(decoder)
        ret = np.concatenate((ret, tret))

        return ret

    def expand_decoder(self, decoder):
        ret = {'id': decoder[0]}

        k = 1
        tret = self.ar.expand_decoder(decoder[k:k + self.ar_size])
        for key in tret:
            ret[key] = tret[key]
        k += self.ar_size
        tret = self.td.expand_decoder(decoder[k:k + self.td_size])
        for key in tret:
            ret[key] = tret[key]
        k += self.td_size
        tret = self.rand.expand_decoder(decoder[k:k + self.rand_size])
        for key in tret:
            ret[key] = tret[key]
        k += self.rand_size
        tret = self.nn.expand_decoder(decoder[k:k + self.nn_size])
        for key in tret:
            ret[key] = tret[key]
        k += self.nn_size

        return ret

    def decode(self, decoder):
        decoder = self.expand_decoder(decoder)

        ret = {}
        for i in range(1, len(self.ln)):
            ret['W' + str(i)] = np.zeros(shape=(self.ln[i], self.ln[i - 1]))
            ret['b' + str(i)] = np.zeros(shape=(self.ln[i], 1))

        tret = self.ar.decode(decoder)
        if tret:
            for i in range(1, len(self.ln)):
                ret['W' + str(i)] += tret['W' + str(i)]
                ret['b' + str(i)] += tret['b' + str(i)]
        tret = self.td.decode(decoder)
        if tret:
            for i in range(1, len(self.ln)):
                ret['W' + str(i)] += tret['W' + str(i)]
                ret['b' + str(i)] += tret['b' + str(i)]
        tret = self.rand.decode(decoder)
        if tret:
            for i in range(1, len(self.ln)):
                ret['W' + str(i)] += tret['W' + str(i)]
                ret['b' + str(i)] += tret['b' + str(i)]
        tret = self.nn.decode(decoder)
        if tret:
            for i in range(1, len(self.ln)):
                ret['W' + str(i)] += tret['W' + str(i)]
                ret['b' + str(i)] += tret['b' + str(i)]

        for j in range(1, len(self.ln)):
            ret['func' + str(j)] = self.lfunc[j - 1]

        return ret

    def clip(self, decoder):
        ret = np.array([decoder[0]])

        k = 1
        ret = np.concatenate((ret, self.ar.clip(decoder[k:k + self.ar_size])))
        k += self.ar_size
        ret = np.concatenate((ret, self.td.clip(decoder[k:k + self.td_size])))
        k += self.td_size
        ret = np.concatenate((ret, self.rand.clip(decoder[k:k + self.rand_size])))
        k += self.rand_size
        ret = np.concatenate((ret, self.nn.clip(decoder[k:k + self.nn_size])))
        k += self.nn_size

        return ret

    def tf_model(self, weights):
        ret = tf.keras.Sequential()
        ret.add(tf.keras.layers.InputLayer(input_shape=(20, 20, 2)))
        ret.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
        ret.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
        ret.add(tf.keras.layers.Conv2D(filters=4, kernel_size=3, activation='relu', padding='same'))
        ret.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

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
                J = self.problem.test(self.decode(pop[p]))[0]
                fitness[p] = J
            pop = pop[np.argsort(-fitness)]
            fitness = fitness[np.argsort(-fitness)]

            J, steps, score, game_score = self.problem.test(self.decode(pop[0]))
            i = pop[0][0]
            # if t % 25 == 0:
            print('Iter ' + str(t).zfill(3) + '   ' + str(int(i)).zfill(6) + '   ' +
                  str('{:.2f}'.format(J)).zfill(7) + '   ' + str('{:.2f}'.format(steps)).zfill(7) +
                  '   ' + str('{:.2f}'.format(score)).zfill(6) + '   ' + str('{:.2f}'.format(game_score)).zfill(7))

            prob = np.array(fitness, copy=True) - min(0, np.amin(fitness))  # prevent negatives
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

                pop[i][0] = self.count
                self.count += 1
                pop[i + 1][0] = self.count
                self.count += 1

        model = self.decode(pop[0])
        _, steps, score, game_score = self.problem.test(model, 1000)
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6) + '   ' + str('{:.2f}'.format(game_score)).zfill(7))
        print('==================================================')
        i = input()
        self.problem.test(model, 1, 30, gui=True)
        # np.savez('saves/egaSave-' + self.file_name + '.npz',
        #          W1=model['W1'], b1=model['b1'], W2=model['W2'], b2=model['b2'])

    # def test(self, file):
    #     f = np.load('saves/egaSave-' + file + '.npz')
        # model = {'W1': f['W1'], 'b1': f['b1'], 'W2': f['W2'], 'b2': f['b2']}
        # self.problem.test(model, 1000)
