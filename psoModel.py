
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
import random
import pprint


class PSO:
    def __init__(self, problem, td_N, iterations=100,
                 run_name='', ckpt_period=25, load_info=['False', '_', '_'],
                 pop_size=200, W=0.8, c1=1, c2=1):

        # Initializing globals
        self.iterations = iterations
        self.pop_size = pop_size
        self.velocity = [None] * self.pop_size
        self.global_best = {'fitness': None, 'decoder': None}
        self.personal_best = {'fitness': [None] * self.pop_size,
                              'decoder': [None] * self.pop_size}
        self.W = W
        self.c1 = c1
        self.c2 = c2

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
        if run_name != '':
            self.run_name = 'saves/pso-' + self.problem_name + \
                '-' + run_name
            if not os.path.isdir(self.run_name):
                os.mkdir(self.run_name)
        else:
            print('Decaying mutation rate will be disabled')
            self.run_name = ''
        self.ckpt_period = ckpt_period
        self.load_ckpt = load_info[0] == 'True'
        self.load_name = 'saves/pso-' + self.problem_name + '-' + \
            load_info[1]
        self.load_iter = int(load_info[2])

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
            self.layer_shapes = [[[28, 28, 16, 32], [0]]]
            self.layer_sizes = [[28 * 28 * 16 * 32, 0]]

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
            self.td_sizes = [[[28, 28, 16, 32], [0, 0, 0]]]

        self.decoder = ETD(td_N, self.td_sizes, self.layer_shapes, self.layer_sizes)

        # Initializing population variables
        self.count = 0
        self.compress_len = 1 + self.decoder.size

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

    def get_decoder(self, custom_data=False):
        ret = self.decoder.get_decoder(custom_data)
        ret['id'] = self.count
        self.count += 1

        # ret = {'id': self.count}
        # self.count += 1

        # for e in self.decoder_methods:
        #     tret = e.decoder(custom_data)
        #     for key in tret:
        #         ret[key] = tret[key]

        return ret

    def compress_decoder(self, decoder):
        ret = self.decoder.compress_decoder(decoder)
        ret = np.concatenate((np.array([decoder['id']]), ret))

        # ret = np.array([decoder['id']])
        # for e in self.decoder_methods:
        #     tret = e.compress_decoder(decoder)
        #     ret = np.concatenate((ret, tret))

        return ret

    def expand_decoder(self, decoder):
        ret = self.decoder.expand_decoder(decoder[1:])
        ret['id'] = decoder[0]

        # ret = {'id': decoder[0]}
        # decoder = decoder[1:]
        # for e in self.decoder_methods:
        #     tret, decoder = e.expand_decoder(decoder)
        #     for key in tret:
        #         ret[key] = tret[key]

        return ret

    def decode(self, decoder):
        decoder = self.expand_decoder(decoder)
        ret = self.decoder.decode(decoder)

        # decoder = self.expand_decoder(decoder)
        # ret = {}
        # for i in range(len(self.layers)):
        #     ret['W' + str(i)] = np.zeros(shape=self.layer_shapes[i][0])
        #     ret['b' + str(i)] = np.zeros(shape=self.layer_shapes[i][1])
        #     ret['func' + str(i)] = self.layers[i][-1]

        # for e in self.decoder_methods:
        #     tret = e.decode(decoder)
        #     for i in range(len(self.layers)):
        #         ret['W' + str(i)] += tret['W' + str(i)]
        #         ret['b' + str(i)] += tret['b' + str(i)]

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
        ret = self.decoder.clip(decoder[1:])
        ret = np.concatenate((np.array([decoder[0]], copy=True), ret))

        # ret = np.array([decoder[0]], copy=True)
        # decoder = decoder[1:]
        # for e in self.decoder_methods:
        #     tret, decoder = e.clip(decoder)
        #     ret = np.concatenate((ret, tret))

        return ret

    def move(self, decoder, velocity, personal_best):
        ret = decoder[1:].copy()
        v = self.W * velocity + \
            self.c1 * random.random() * (personal_best - decoder[1:]) + \
            self.c2 * random.random() * (self.global_best['decoder'] - decoder[1:])
        ret = ret + v
        ret = np.concatenate((np.array([decoder[0]], copy=True), ret))

        return ret, v

    def update_params(self, t):
        N = self.iterations
        self.W = (0.4 / N ** 2) * (t - N) ** 2 + 0.4
        self.c1 = -3 * t / N + 3.5
        self.c2 = 3 * t / N + 0.5

    def initialize_pop(self):
        if self.load_ckpt:
            self.pop = np.load(self.load_name + '/iter-' + str(self.load_iter) + '.npy')
            if self.problem_name[0:5] == 'snake':
                self.food_arr = np.load(self.load_name + '/food.npy').tolist()
            print('Checkpoint loaded')
        else:
            self.pop = np.array([np.zeros(self.compress_len)] * self.pop_size)
            for p in range(self.pop_size):
                self.pop[p] = self.compress_decoder(self.get_decoder())
                self.pop[p] = self.clip(self.pop[p])
                self.velocity[p] = (np.random.random(self.compress_len - 1) - 0.5) / 100
            if self.problem_name[0:5] == 'snake':
                randx = np.random.randint(1, 20, 3000)
                randy = np.random.randint(1, 20, 3000)
                self.food_arr = [[i, j] for i, j in zip(randx, randy)]
                if self.run_name != '':
                    np.save(self.run_name + '/food.npy', self.food_arr)
        if self.load_iter != -1:
            self.start_iter = self.load_iter + 1
        else:
            self.start_iter = 1

    def run(self):

        # Preparing records and files
        if self.run_name != '':
            history = open(self.run_name + '/hist.txt', 'a+')
            params = open(self.run_name + '/params.txt', 'a+')
            values = self.__dict__.copy()
            values.pop('pop')
            params.write(pprint.pformat(values))
            params.write('\n')
            params.close()

        if self.problem_name[0:5] == 'snake':
            model = self.get_tf_model()
        if self.problem_name == 'MNIST':
            model = self.get_tf_model()
        if self.problem_name == 'weights':
            pool = mp.Pool(4)

        for t in range(self.start_iter, self.start_iter + self.iterations + 1):
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
                    model = self.set_tf_weights(model, self.decode(self.pop[p]))
                    steps[p], score[p], art_score[p], move_distributions[p], _, _, _ \
                        = self.problem.test(model, food_arr=self.food_arr)
                    fitness[p] = art_score[p]
                    if self.global_best['fitness'] is None or \
                            fitness[p] > self.global_best['fitness']:
                        self.global_best['decoder'] = self.pop[p][1:]
                    if self.personal_best['fitness'][p] is None or \
                            fitness[p] > self.personal_best['fitness'][p]:
                        self.personal_best['decoder'][p] = self.pop[p][1:]
            if self.problem_name == 'MNIST':
                for p in range(self.pop_size):
                    model = self.set_tf_weights(model, self.decode(self.pop[p]))
                    cce[p], acc[p] = self.problem.test(model)
                    fitness[p] = cce[p]
                    if self.global_best['fitness'] is None or \
                            fitness[p] < self.global_best['fitness']:
                        self.global_best['decoder'] = self.pop[p][1:]
                    if self.personal_best['fitness'][p] is None or \
                            fitness[p] < self.personal_best['fitness'][p]:
                        self.personal_best['decoder'][p] = self.pop[p][1:]
            if self.problem_name == 'weights':
                diff = pool.map(self.problem.test,
                                [self.decode(ind)['W0'] for ind in self.pop])
                diff = np.array(diff)
                fitness = diff
                for p in range(self.pop_size):
                    if self.global_best['fitness'] is None or \
                            fitness[p] < self.global_best['fitness']:
                        self.global_best['decoder'] = self.pop[p][1:]
                    if self.personal_best['fitness'][p] is None or \
                            fitness[p] < self.personal_best['fitness'][p]:
                        self.personal_best['decoder'][p] = self.pop[p][1:]

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
            self.pop = self.pop[sort]
            fitness = fitness[sort]

            # Moving particles
            for i in range(self.pop_size):
                self.pop[i], self.velocity[i] = self.move(self.pop[i], self.velocity[i], self.personal_best['decoder'][i])

            self.update_params(t)

            # Recording performance
            if self.run_name != '':
                if self.problem_name[0:5] == 'snake':
                    avg_steps = 1.0 * np.sum(steps) / self.pop_size
                    avg_score = 1.0 * np.sum(score) / self.pop_size
                    avg_art_score = 1.0 * np.sum(art_score) / self.pop_size
                    history.write(str(t).zfill(6) + '     ' +
                                  str(int(self.pop[0][0])).zfill(6) + '   ' +
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
                if self.run_name != '':
                    np.save(self.run_name + '/iter-' + str(t) + '.npy', self.pop)

                if self.problem_name[0:5] == 'snake':
                    print(str(t).zfill(6) + '   ' +
                          str(int(self.pop[0][0])).zfill(6) + '   ' +
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

        if self.run_name != '':
            history.close()
        if self.problem_name == 'weights':
            pool.close()

        # Saving population into files
        if self.run_name != '':
            if self.problem_name[0:5] == 'snake':
                np.savez(self.run_name + '/final.npz',
                         iterations=self.iterations,
                         pop=self.pop,
                         food=self.food_arr)
            if self.problem_name == 'MNIST':
                np.savez(self.run_name + '/final.npz',
                         iterations=self.iterations,
                         pop=self.pop)
            if self.problem_name == 'weights':
                np.savez(self.run_name + '/final.npz',
                         iterations=self.iterations,
                         pop=self.pop)
        return self.test()

    def test(self):
        # Load in pop
        if self.load_ckpt:
            self.pop = np.load(self.load_name + '/iter-' + str(self.load_iter) + '.npy')
            if self.problem_name[0:5] == 'snake':
                food_arr = np.load(self.load_name + '/food.npy').tolist()

        weights = self.decode(self.pop[0])
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
