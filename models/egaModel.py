
import sys
sys.path.insert(0, '../problems')

from scoreProblem import ScoreProb
from actionProblem import ActionProb

from eAR import EAR
from eTD import ETD
from eRand import ERand
from eNN import ENN

import os
import numpy as np
import tensorflow as tf
import time


class EGA:
    def __init__(self, problem, ar_N, td_N, rand_N, nn_N, iterations=100,
                 pop_size=200, mut_prob=0.3, elite_ratio=0.01, cross_prob=0.7,
                 par_ratio=0.3, file_name='default', ckpt_period=25):
        self.iterations = iterations
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)
        self.file_name = file_name
        self.folder_name = 'saves/egaSave-' + self.file_name
        if not os.path.isdir(self.folder_name):
            os.mkdir(self.folder_name)
        self.ckpt_period = ckpt_period

        self.problem = None
        if problem == 'Score':
            self.problem = ScoreProb()
        if problem == 'Action':
            self.problem = ActionProb()

        self.ln = [[24, 24, 6], [5, 5, 6, 32], [3, 3, 32, 16], [3, 3, 16, 8],
                   [288, 128], [128, 128], [128, 64], [64, 64],
                   [64, 16], [16, 16], [16, 4]]

        self.lfunc = ['relu', 'pool', 'relu', 'pool', 'relu', 'flatten',
                      'relu', 'relu', 'relu', 'relu',
                      'relu', 'relu', 'linear']

        self.count = 0

        self.td_N = td_N
        self.ar_N = ar_N
        self.rand_N = rand_N
        self.nn_N = nn_N

        self.td_sizes = [[10, 22, 22], [16, 17, 17], [6, 13, 15],
                         [32, 34, 34], [16, 24, 43], [17, 18, 27], [13, 16, 20],
                         [8, 10, 13], [3, 7, 13], [2, 5, 7]]

        self.eln1 = 3
        self.eln2 = 4
        self.eln3 = 1

        self.ar_size = (2 * len(self.lfunc)) * self.ar_N
        self.td_size = (np.sum(self.td_sizes) + len(self.td_sizes)) * self.td_N
        self.rand_size = 2 * self.rand_N
        self.nn_size = (self.eln2 * self.eln1 + self.eln2 +
                        self.eln3 * self.eln2 + self.eln3) * self.nn_N

        self.ar = EAR(self.ar_N, self.ln)
        self.td = ETD(self.td_N, self.td_sizes, self.ln)
        self.rand = ERand(self.rand_N, self.ln)
        self.nn = ENN(self.nn_N, self.eln1, self.eln2, self.eln3, self.ln)
        self.compress_len = self.compress_decoder(self.decoder()).size

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
            ret['W' + str(i)] = np.zeros(shape=self.ln[i])
            ret['b' + str(i)] = np.zeros(shape=(self.ln[i][-1]))

#         tret = self.ar.decode(decoder)
#         if tret:
#             for i in range(1, len(self.lfunc)):
#                 ret['W' + str(i)] += tret['W' + str(i)]
#                 ret['b' + str(i)] += tret['b' + str(i)]
        tret = self.td.decode(decoder)
        if tret:
            for i in range(1, len(self.ln)):
                ret['W' + str(i)] += tret['W' + str(i)]
                ret['b' + str(i)] += tret['b' + str(i)]
#         tret = self.rand.decode(decoder)
#         if tret:
#             for i in range(1, len(self.lfunc)):
#                 ret['W' + str(i)] += tret['W' + str(i)]
#                 ret['b' + str(i)] += tret['b' + str(i)]
#         tret = self.nn.decode(decoder)
#         if tret:
#             for i in range(1, len(self.lfunc)):
#                 ret['W' + str(i)] += tret['W' + str(i)]
#                 ret['b' + str(i)] += tret['b' + str(i)]

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

    def get_tf_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=self.ln[0]))
        j = 1
        for i in range(0, len(self.lfunc)):
            if self.lfunc[i] == 'pool':
                model.add(tf.keras.layers.MaxPool2D(2, 2, padding='valid'))
            elif self.lfunc[i] == 'flatten':
                model.add(tf.keras.layers.Flatten())
            else:
                if len(self.ln[j]) > 2:
                    model.add(tf.keras.layers.Conv2D(self.ln[j][-1], self.ln[j][0], activation=self.lfunc[i], padding='same'))
                else:
                    model.add(tf.keras.layers.Dense(self.ln[j][-1], activation=self.lfunc[i]))
                j += 1
        return model

    def set_tf_weights(self, model, weights):
        j = 1
        for i in range(0, len(self.lfunc)):
            if self.lfunc[i] == 'pool' or self.lfunc[i] == 'flatten':
                pass
            else:
                model.layers[i].set_weights([weights['W' + str(j)], weights['b' + str(j)]])
                j += 1
        return model

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
                c[i] += np.random.normal(scale=1)
        c = self.clip(c)
        return c

    def run(self):
        history = open(self.folder_name + '/hist.txt', 'w+')

        foo = input('Load previous checkpoint? (y/n) ')
        if foo == 'y':
            load_name = input('Save name: ')
            start_iter = int(input('Iteration: '))
            pop = np.load('saves/egaSave-' + load_name + '/iter-' + str(start_iter) + '.npy')
            food_arr = np.load('saves/egaSave-' + load_name + '/food.npy').tolist()
        else:
            start_iter = 1
            pop = np.array([np.zeros(self.compress_len)] * self.pop_size)
            for p in range(self.pop_size):
                pop[p] = self.compress_decoder(self.decoder())
                pop[p] = self.clip(pop[p])
            randx = np.random.randint(1, 20, 3000)
            randy = np.random.randint(1, 20, 3000)
            food_arr = [[i, j] for i, j in zip(randx, randy)]  # Test with preset food positions
        np.save(self.folder_name + '/food.npy', food_arr)

        model = self.get_tf_model()
        for t in range(start_iter, self.iterations + 1):
            start = time.time()
            fitness = np.zeros(self.pop_size)
            steps = np.zeros(self.pop_size)
            # games = np.zeros(self.pop_size)
            score = np.zeros(self.pop_size)
            art_score = np.zeros(self.pop_size)
            move_distributions = np.zeros((self.pop_size, 4))

            for p in range(self.pop_size):
                model = self.set_tf_weights(model, self.decode(pop[p]))
                steps[p], score[p], art_score[p], move_distributions[p], _, _, _ = self.problem.test_over_games(model, food_arr=food_arr)
                fitness[p] = art_score[p]
            sort = np.argsort(-fitness)
            pop = pop[sort]
            fitness = fitness[sort]
            steps = steps[sort]
            # games = games[sort]
            score = score[sort]
            art_score = art_score[sort]
            move_distributions = move_distributions[sort]

            avg_steps = 1.0 * np.sum(steps) / self.pop_size
            avg_score = 1.0 * np.sum(score) / self.pop_size
            avg_art_score = 1.0 * np.sum(art_score) / self.pop_size
            history.write(str(t).zfill(6) + '     ' +
                          str(int(pop[0][0])).zfill(6) + '   ' +
                          str('{:.2f}'.format(steps[0])).zfill(7) + '   ' +
                          str('{:.2f}'.format(score[0])).zfill(6) + '   ' +
                          str('{:.2f}'.format(art_score[0])).zfill(7) + '     ' +
                          str('{:.2f}'.format(avg_steps)).zfill(7) + '   ' +
                          str('{:.2f}'.format(avg_score)).zfill(6) + '   ' +
                          str('{:.2f}'.format(avg_art_score)).zfill(7) + '\n')
            history.flush()

            end = time.time()
            if t % self.ckpt_period == 0:
                np.save(self.folder_name + '/iter-' + str(t) + '.npy', pop)
                prev_file = self.folder_name + '/iter-' + str(t - 10 * self.ckpt_period) + '.npy'
                if os.path.exists(prev_file):
                    os.remove(prev_file)

                print('Iter ' + str(t).zfill(6) + '   ' +
                      str(int(pop[0][0])).zfill(6) + '   ' +
                      str('{:.2f}'.format(steps[0])).zfill(7) + '   ' +
                      # str('{:.2f}'.format(games[0])).zfill(5) + '   ' +
                      str('{:.2f}'.format(score[0])).zfill(6) + '   ' +
                      str('{:.2f}'.format(art_score[0])).zfill(7) + '     ' +
                      str(move_distributions[0][0]).zfill(5) + '   ' +
                      str(move_distributions[0][1]).zfill(5) + '   ' +
                      str(move_distributions[0][2]).zfill(5) + '   ' +
                      str(move_distributions[0][3]).zfill(5) + '          ' +
                      str('{:.2f}'.format(end - start).zfill(6)) + 's')

            # if t % 50 == 0:
            #     self.problem.test_over_games(self.set_tf_weights(model, self.decode(pop[0])), goal_steps=30, food_arr=food_arr, gui=True)

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

        history.close()
        np.savez(self.folder_name + '/final.npz', iterations=self.iterations, pop=pop, food=food_arr)
        self.test(pop, food_arr)

    def test(self, pop, food_arr):
        weights = self.decode(pop[0])
        model = self.get_tf_model()
        model = self.set_tf_weights(model, weights)
        print('==================================================')
        steps, score, art_score, move_distributions, steps_arr, score_arr, art_score_arr = self.problem.test_over_games(model, test_games=1, food_arr=food_arr)
        print('Results Over Preset Food:   ' +
              str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6) + '   ' +
              str('{:.2f}'.format(art_score)).zfill(7))
        steps, score, art_score, move_distributions, steps_arr, score_arr, art_score_arr = self.problem.test_over_games(model, test_games=100)
        print('Results Over Random Food:   ' +
              str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6) + '   ' +
              str('{:.2f}'.format(art_score)).zfill(7))
        sort = np.argsort(-steps_arr)
        steps_arr = steps_arr[sort]
        score_arr = score_arr[sort]
        art_score_arr = art_score_arr[sort]
        print('              Best Steps:   ' +
              str('{:.2f}'.format(steps_arr[0])).zfill(7) + '   ' +
              str('{:.2f}'.format(score_arr[0])).zfill(6) + '   ' +
              str('{:.2f}'.format(art_score_arr[0])).zfill(7))
        sort = np.argsort(-score_arr)
        steps_arr = steps_arr[sort]
        score_arr = score_arr[sort]
        art_score_arr = art_score_arr[sort]
        print('              Best Score:   ' +
              str('{:.2f}'.format(steps_arr[0])).zfill(7) + '   ' +
              str('{:.2f}'.format(score_arr[0])).zfill(6) + '   ' +
              str('{:.2f}'.format(art_score_arr[0])).zfill(7))
        print('==================================================')
        input()
        self.problem.test_over_games(model, goal_steps=30, gui=True)
