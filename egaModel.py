
import sys
sys.path.insert(0, '../problems')

from snakeScore import SnakeScore
from snakeAction import SnakeAction
from mnist import MNIST
from weights import Weights

from eAR import EAR
from eTD import ETD
from eRand import ERand
from eNN import ENN

import os
import numpy as np
import tensorflow as tf
import time
import multiprocessing as mp


class EGA:
    def __init__(self, problem, ar_N, td_N, rand_N, nn_N, iterations=100,
                 pop_size=200, mut_prob=0.3, elite_ratio=0.01, cross_prob=0.7,
                 par_ratio=0.3, folder_name='default', ckpt_period=25,
                 load_ckpt=False, load_name='null', load_iter='-1',
                 td_mut_scale=0.001):

        # Initializing globals
        self.iterations = iterations
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)
        self.td_mut_scale = td_mut_scale

        # Setting problem
        self.problem = None
        if problem == 'snake_score':
            self.problem = SnakeScore()
        if problem == 'snake_action':
            self.problem = SnakeAction()
        if problem == 'MNIST':
            self.problem = MNIST()
        if problem == 'weights':
            self.problem = Weights()
        self.problem_name = problem

        # IO
        self.folder_name = 'saves/ega-' + self.problem_name + \
            '-' + folder_name
        if not os.path.isdir(self.folder_name) and self.folder_name != 'print':
            os.mkdir(self.folder_name)
        self.ckpt_period = ckpt_period
        self.load_ckpt = load_ckpt
        self.load_name = 'saves/ega-' + self.problem_name + '-' + load_name
        self.load_iter = load_iter

        ##################################################
        # NETWORK ARCHITECTURE
        ##################################################
        if self.problem_name[0:5] == 'snake':
            self.layers = [['input', 24, 24, 6],
                           ['conv', 5, 32, 'relu', 'same'],
                           ['pool'],
                           ['conv', 3, 16, 'relu', 'same'],
                           ['pool'],
                           ['conv', 3, 8, 'relu', 'save'],
                           ['flatten'],
                           ['dense', 128, 'relu'],
                           ['dense', 128, 'relu'],
                           ['dense', 64, 'relu'],
                           ['dense', 64, 'relu'],
                           ['dense', 16, 'relu'],
                           ['dense', 16, 'relu'],
                           ['dense', 4, 'linear']]
            self.layer_shapes, self.layer_sizes = self.get_layer_info()
        if self.problem_name == 'MNIST':
            self.layers = [['input', 28, 28, 1],
                           ['conv', 5, 6, 'tanh', 'same'],
                           ['pool'],
                           ['conv', 5, 16, 'tanh', 'valid'],
                           ['pool'],
                           ['flatten'],
                           ['dense', 120, 'tanh'],
                           ['dense', 84, 'tanh'],
                           ['dense', 10, 'linear']]
            self.layer_shapes, self.layer_sizes = self.get_layer_info()
        if self.problem_name == 'weights':
            self.layers = [['null']]
            # self.layer_shapes = [[[28, 28, 16, 32], [0]]]
            # self.layer_sizes = [[28 * 28 * 16 * 32, 0]]
            self.layer_shapes = [[[28, 28], [0]]]
            self.layer_sizes = [[784, 0]]

        ##################################################
        # DECODERS
        ##################################################
        if self.problem_name[0:5] == 'snake':
            self.td_sizes = [[],
                             [[12, 20, 20], [2, 4, 4]],
                             [],
                             [[16, 16, 18], [2, 2, 4]],
                             [],
                             [[8, 9, 16], [2, 2, 2]],
                             [],
                             [[32, 32, 36], [4, 4, 8]],
                             [[16, 32, 32], [4, 4, 8]],
                             [[16, 16, 32], [4, 4, 4]],
                             [[16, 16, 16], [4, 4, 4]],
                             [[8, 8, 16], [2, 2, 4]],
                             [[4, 8, 8], [2, 2, 4]],
                             [[4, 4, 4], [1, 2, 2]]]
        if self.problem_name == 'MNIST':
            self.td_sizes = [[],
                             [[5, 5, 6], [1, 2, 3]],
                             [],
                             [[10, 15, 16], [2, 2, 4]],
                             [],
                             [],
                             [[30, 40, 40], [4, 5, 6]],
                             [[18, 20, 28], [3, 4, 7]],
                             [[7, 10, 12], [1, 2, 5]]]
        if self.problem_name == 'weights':
            # self.td_sizes = [[[28, 28, 16, 32], [0, 0, 0]]]
            self.td_sizes = [[[28, 28], [0]]]

        self.decoder_methods = []
        if ar_N > 0:  # Outdated
            self.decoder_methods.append(EAR(ar_N, self.layer_shapes,
                                            self.layer_sizes))
        if td_N > 0:
            self.decoder_methods.append(ETD(td_N, self.td_sizes, self.layer_shapes,
                                            self.layer_sizes, self.td_mut_scale))
        if rand_N > 0:  # Outdated
            self.decoder_methods.append(ERand(rand_N, self.layer_shapes,
                                              self.layer_sizes))
        if nn_N > 0:  # Outdated
            self.decoder_methods.append(ENN(nn_N, self.layer_shapes,
                                            self.layer_sizes))

        # Initializing population variables
        self.count = 0
        self.compress_len = 1
        for e in self.decoder_methods:
            self.compress_len += e.size

    def get_layer_info(self):
        sizes_ret = [[0, 0]]
        shapes_ret = [[[0], [0]]]
        cur = self.layers[0][1:]
        for i in range(1, len(self.layers)):
            layer = self.layers[i]
            if layer[0] == 'conv':
                sizes_ret.append([layer[1] * layer[1] *
                                  cur[2] * layer[2],
                                  layer[2]])
                shapes_ret.append([[layer[1], layer[1],
                                    cur[2], layer[2]],
                                   [layer[2]]])
                cur[2] = layer[2]
                if layer[4] == 'valid':
                    cur[0] -= (layer[1] - 1)
                    cur[1] -= (layer[1] - 1)
            if layer[0] == 'pool':
                sizes_ret.append([0, 0])
                shapes_ret.append([[0], [0]])
                cur[0] /= 2
                cur[1] /= 2
            if layer[0] == 'flatten':
                sizes_ret.append([0, 0])
                shapes_ret.append([[0], [0]])
                cur = [int(cur[0] * cur[1] * cur[2])]
            if layer[0] == 'dense':
                sizes_ret.append([layer[1] * cur[0], layer[1]])
                shapes_ret.append([[cur[0], layer[1]], [layer[1]]])
                cur = [layer[1]]

        return shapes_ret, sizes_ret

    def decoder(self):
        ret = {'id': self.count}
        self.count += 1

        for e in self.decoder_methods:
            tret = e.decoder()
            for key in tret:
                ret[key] = tret[key]

        return ret

    def compress_decoder(self, decoder):
        ret = np.array([decoder['id']])
        for e in self.decoder_methods:
            tret = e.compress_decoder(decoder)
            ret = np.concatenate((ret, tret))

        return ret

    def expand_decoder(self, decoder):
        ret = {'id': decoder[0]}
        decoder = decoder[1:]
        for e in self.decoder_methods:
            tret, decoder = e.expand_decoder(decoder)
            for key in tret:
                ret[key] = tret[key]

        return ret

    def decode(self, decoder):
        decoder = self.expand_decoder(decoder)

        ret = {}
        for i in range(len(self.layers)):
            ret['W' + str(i)] = np.zeros(shape=self.layer_shapes[i][0])
            ret['b' + str(i)] = np.zeros(shape=self.layer_shapes[i][1])
            ret['func' + str(i)] = self.layers[i][-1]

        for e in self.decoder_methods:
            tret = e.decode(decoder)
            for i in range(len(self.layers)):
                ret['W' + str(i)] += tret['W' + str(i)]
                ret['b' + str(i)] += tret['b' + str(i)]

        return ret

    def get_tf_model(self):
        model = tf.keras.models.Sequential()
        for i in range(0, len(self.layers)):
            if self.layers[i][0] == 'input':
                model.add(tf.keras.layers.InputLayer(input_shape=self.layers[i][1:]))
            if self.layers[i][0] == 'conv':
                model.add(tf.keras.layers.Conv2D(self.layers[i][2], self.layers[i][1],
                                                 activation=self.layers[i][3], padding=self.layers[i][4]))
            if self.layers[i][0] == 'pool':
                model.add(tf.keras.layers.MaxPool2D(2, 2, padding='valid'))
            if self.layers[i][0] == 'flatten':
                model.add(tf.keras.layers.Flatten())
            if self.layers[i][0] == 'dense':
                model.add(tf.keras.layers.Dense(self.layers[i][1], activation=self.layers[i][2]))
        if self.problem_name == 'MNIST':
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])

        return model

    def set_tf_weights(self, model, weights, copy=False):
        if copy:
            ret = tf.keras.models.clone_model(model)
        else:
            ret = model
        for i in range(0, len(self.layers)):
            if self.layers[i][0] == 'input' or \
                    self.layers[i][0] == 'pool' or \
                    self.layers[i][0] == 'flatten':
                continue
            ret.layers[i - 1].set_weights([weights['W' + str(i)], weights['b' + str(i)]])

        return ret

    def clip(self, decoder):
        ret = np.array([decoder[0]], copy=True)
        decoder = decoder[1:]
        for e in self.decoder_methods:
            tret, decoder = e.clip(decoder)
            ret = np.concatenate((ret, tret))

        return ret

    def cross(self, x, y):
        c1 = x.copy()
        c2 = y.copy()
        for i in range(1, self.compress_len):
            if np.random.random() < 0.5:
                c1[i] = y[i].copy()
                c2[i] = x[i].copy()
        c1 = self.clip(c1)
        c2 = self.clip(c2)

        return c1, c2

    def mut(self, decoder):
        ret = np.array([decoder[0]], copy=True)
        for e in self.decoder_methods:
            tret, decoder = e.mut(decoder[1:], self.mut_prob)
            ret = np.concatenate((ret, tret))
        return ret

    def run(self):

        # Preparing records and files
        history = open(self.folder_name + '/hist.txt', 'a+')

        # Initializing population and problem-specific variables
        if self.load_ckpt:
            pop = np.load(self.load_name + '/iter-' + str(self.load_iter) + '.npy')
            if self.problem_name[0:5] == 'snake':
                food_arr = np.load(self.load_name + '/food.npy').tolist()
            start_iter = self.load_iter + 1
        else:
            start_iter = 1
            pop = np.array([np.zeros(self.compress_len)] * self.pop_size)
            for p in range(self.pop_size):
                pop[p] = self.compress_decoder(self.decoder())
                pop[p] = self.clip(pop[p])
            if self.problem_name[0:5] == 'snake':
                randx = np.random.randint(1, 20, 3000)
                randy = np.random.randint(1, 20, 3000)
                food_arr = [[i, j] for i, j in zip(randx, randy)]
                np.save(self.folder_name + '/food.npy', food_arr)
        if self.problem_name[0:5] == 'snake':
            model = self.get_tf_model()
        if self.problem_name == 'MNIST':
            model = self.get_tf_model()
        if self.problem_name == 'weights':
            pool = mp.Pool(4)

        for t in range(start_iter, start_iter + self.iterations + 1):
            start = time.time()

            # Initializing metrics
            fitness = np.zeros(self.pop_size)
            if self.problem_name[0:5] == 'snake':
                steps = np.zeros(self.pop_size)
                score = np.zeros(self.pop_size)
                art_score = np.zeros(self.pop_size)
                move_distributions = np.zeros((self.pop_size, 4))
            if self.problem_name == 'MNIST':
                cce = np.zeros(self.pop_size)
                acc = np.zeros(self.pop_size)
            if self.problem_name == 'weights':
                diff = np.zeros(self.pop_size)

            # Testing population
            if self.problem_name[0:5] == 'snake':
                for p in range(self.pop_size):
                    model = self.set_tf_weights(model, self.decode(pop[p]))
                    steps[p], score[p], art_score[p], move_distributions[p], _, _, _ \
                        = self.problem.test(model, food_arr=food_arr)
                    fitness[p] = art_score[p]
            if self.problem_name == 'MNIST':
                for p in range(self.pop_size):
                    model = self.set_tf_weights(model, self.decode(pop[p]))
                    cce[p], acc[p] = self.problem.test(model)
                    fitness[p] = cce[p]
            if self.problem_name == 'weights':
                diff = pool.map(self.problem.test,
                                [self.decode(ind)['W0'] for ind in pop])
                # diff[p] = self.problem.test(self.decode(pop[p])['W0'])
                # fitness[p] = diff[p]
                diff = np.array(diff)
                fitness = diff

            # Sorting population
            if self.problem_name[0:5] == 'snake':
                sort = np.argsort(-fitness)
                steps = steps[sort]
                score = score[sort]
                art_score = art_score[sort]
                move_distributions = move_distributions[sort]
            if self.problem_name == 'MNIST':
                sort = np.argsort(fitness)
                cce = cce[sort]
                acc = acc[sort]
            if self.problem_name == 'weights':
                sort = np.argsort(fitness)
                diff = diff[sort]
            pop = pop[sort]
            fitness = fitness[sort]

            # Recording performance
            if self.problem_name[0:5] == 'snake':
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
            if self.problem_name == 'MNIST':
                avg_cce = 1.0 * np.sum(cce) / self.pop_size
                avg_acc = 1.0 * np.sum(acc) / self.pop_size
                history.write(str(t).zfill(6) + '     ' +
                              str('{:.2f}'.format(cce[0])).zfill(3) + '   ' +
                              str('{:.2f}'.format(acc[0])).zfill(5) + '   ' +
                              str('{:.2f}'.format(avg_cce)).zfill(3) + '   ' +
                              str('{:.2f}'.format(avg_acc)).zfill(5) + '\n')
            if self.problem_name == 'weights':
                avg_diff = 1.0 * np.sum(diff) / self.pop_size
                history.write(str(t).zfill(6) + '     ' +
                              str('{:.2f}'.format(diff[0])).zfill(3) + '   ' +
                              str('{:.2f}'.format(avg_diff)).zfill(5) + '\n')
            history.flush()

            end = time.time()

            # Reporting information
            if t % self.ckpt_period == 0:
                np.save(self.folder_name + '/iter-' + str(t) + '.npy', pop)
                # prev_file = self.folder_name + '/iter-' + str(t - 10 * self.ckpt_period) + '.npy'
                # if os.path.exists(prev_file):
                #     os.remove(prev_file)

                if self.problem_name[0:5] == 'snake':
                    print(str(t).zfill(6) + '   ' +
                          str(int(pop[0][0])).zfill(6) + '   ' +
                          str('{:.2f}'.format(steps[0])).zfill(7) + '   ' +
                          str('{:.2f}'.format(score[0])).zfill(6) + '   ' +
                          str('{:.2f}'.format(art_score[0])).zfill(7) + '     ' +
                          str(move_distributions[0][0]).zfill(5) + '   ' +
                          str(move_distributions[0][1]).zfill(5) + '   ' +
                          str(move_distributions[0][2]).zfill(5) + '   ' +
                          str(move_distributions[0][3]).zfill(5) + '          ' +
                          str('{:.2f}'.format(end - start).zfill(6)) + 's')
                if self.problem_name == 'MNIST':
                    print(str(t).zfill(6) + '     ' +
                          str('{:.2f}'.format(cce[0])).zfill(3) + '   ' +
                          str('{:.2f}'.format(acc[0])).zfill(5) + '   ' +
                          str('{:.2f}'.format(end - start).zfill(6)) + 's')
                if self.problem_name == 'weights':
                    print(str(t).zfill(6) + '     ' +
                          str('{:.2f}'.format(diff[0])).zfill(3) + '   ' +
                          str('{:.2f}'.format(end - start).zfill(6)) + 's')

            # Calculating survival probabilities
            prob = np.array(fitness, copy=True)
            if self.problem_name[0:5] == 'snake':  # Bigger is better
                continue
            if self.problem_name == 'MNIST':  # Smaller is better
                prob = np.amax(prob) - prob
            if self.problem_name == 'weights':  # Smaller is better
                prob = np.amax(prob) - prob
            prob = (prob - np.amin(prob)) / (np.amax(prob) - np.amin(prob))
            prob = prob / np.sum(prob)
            cum_prob = np.cumsum(prob)

            # Finding parents
            par = np.array([np.zeros(self.compress_len)] * self.par_size)
            for i in range(self.elite_size):
                par[i] = pop[i].copy()
            for i in range(self.elite_size, self.par_size):
                idx = np.searchsorted(cum_prob, np.random.random())
                par[i] = pop[idx].copy()

            # Performing crossover and mutation
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
                c1, c2 = self.cross(p1, p2)

                pop[i] = c1
                pop[i] = self.mut(pop[i])
                pop[i][0] = self.count
                self.count += 1

                if i + 1 < self.pop_size:
                    pop[i + 1] = c2
                    pop[i + 1] = self.mut(pop[i + 1])
                    pop[i + 1][0] = self.count
                    self.count += 1

        history.close()

        # Saving population into files
        if self.problem_name[0:5] == 'snake':
            np.savez(self.folder_name + '/final.npz', iterations=self.iterations, pop=pop, food=food_arr)
            return self.test(pop, food_arr=food_arr)
        if self.problem_name == 'MNIST':
            np.savez(self.folder_name + '/final.npz', iterations=self.iterations, pop=pop)
            return self.test(pop)
        if self.problem_name == 'weights':
            np.savez(self.folder_name + '/final.npz', iterations=self.iterations, pop=pop)
            return self.test(pop)

    def test(self, pop, food_arr=None):
        weights = self.decode(pop[0])
        print('==================================================')
        if self.problem_name[0:5] == 'snake':
            model = self.get_tf_model()
            model = self.set_tf_weights(model, weights)
            steps, score, art_score, move_distributions, steps_arr, score_arr, art_score_arr = \
                self.problem.test(model, test_games=1, food_arr=food_arr)
            print('Results Over Preset Food:   ' +
                  str('{:.2f}'.format(steps)).zfill(7) + '   ' +
                  str('{:.2f}'.format(score)).zfill(6) + '   ' +
                  str('{:.2f}'.format(art_score)).zfill(7))
            steps, score, art_score, move_distributions, steps_arr, score_arr, art_score_arr = \
                self.problem.test(model, test_games=100)
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
        if self.problem_name == 'MNIST':
            model = self.get_tf_model()
            model = self.set_tf_weights(model, weights)
            cce, acc = self.problem.test(model)
            print('Results:   ' +
                  str('{:.2f}'.format(cce)).zfill(3) + '   ' +
                  str('{:.2f}'.format(acc)).zfill(5))
        if self.problem_name == 'weights':
            diff = self.problem.test(weights['W0'])
            print('Results:   ' +
                  str('{:.2f}'.format(diff)).zfill(3))
        print('==================================================')

        if self.problem_name[0:5] == 'snake':
            input()
            self.problem.test(model, goal_steps=30, gui=True)

        if self.problem_name[0:5] == 'snake':
            return art_score
        if self.problem_name == 'MNIST':
            return cce
        if self.problem_name == 'weights':
            return diff
