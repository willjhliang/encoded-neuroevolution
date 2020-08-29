
import sys
sys.path.insert(0, '../problems')

from snakeScore import SnakeScore
from snakeAction import SnakeAction
from mnist import MNIST

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

        # Initializations
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

        # Setting problem
        self.problem = None
        if problem == 'SnakeScore':
            self.problem = SnakeScore()
        if problem == 'SnakeAction':
            self.problem = SnakeAction()
        if problem == 'MNIST':
            self.problem = MNIST()
        self.problem_name = problem

        # Network architecture
        # self.layers = [['input', 24, 24, 6],
        #                ['conv', 5, 32, 'relu', 'same'],
        #                ['pool'],
        #                ['conv', 3, 16, 'relu', 'same'],
        #                ['pool'],
        #                ['conv', 3, 8, 'relu', 'save'],
        #                ['flatten'],
        #                ['dense', 128, 'relu'],
        #                ['dense', 128, 'relu'],
        #                ['dense', 64, 'relu'],
        #                ['dense', 64, 'relu'],
        #                ['dense', 16, 'relu'],
        #                ['dense', 16, 'relu'],
        #                ['dense', 4, 'linear']]
        self.layers = [['input', 28, 28, 1],  # official LeNet-5 input is 32 x 32
                       ['conv', 5, 6, 'tanh', 'same'],
                       ['pool'],
                       ['conv', 5, 16, 'tanh', 'valid'],
                       ['pool'],
                       ['flatten'],
                       ['dense', 120, 'tanh'],
                       ['dense', 84, 'tanh'],
                       ['dense', 10, 'linear']]

        self.layer_shapes, self.layer_sizes = self.get_layer_info()

        # Setting special parameters for decoders
        # self.td_sizes = [[],
        #                  [[12, 20, 20], [2, 4, 4]],
        #                  [],
        #                  [[16, 16, 18], [2, 2, 4]],
        #                  [],
        #                  [[8, 9, 16], [2, 2, 2]],
        #                  [],
        #                  [[32, 32, 36], [4, 4, 8]],
        #                  [[16, 32, 32], [4, 4, 8]],
        #                  [[16, 16, 32], [4, 4, 4]],
        #                  [[16, 16, 16], [4, 4, 4]],
        #                  [[8, 8, 16], [2, 2, 4]],
        #                  [[4, 8, 8], [2, 2, 4]],
        #                  [[4, 4, 4], [1, 2, 2]]]
        self.td_sizes = [[],
                         [[5, 5, 6], [1, 2, 3]],
                         [],
                         [[10, 15, 16], [2, 2, 4]],
                         [],
                         [],
                         [[30, 40, 40], [4, 5, 6]],
                         [[18, 20, 28], [3, 4, 7]],
                         [[7, 10, 12], [1, 2, 5]]]

        # Creating decoder methods
        self.decoder_methods = []
        if ar_N > 0:  # Outdated
            self.decoder_methods.append(EAR(ar_N, self.layer_shapes,
                                            self.layer_sizes))
        if td_N > 0:
            self.decoder_methods.append(ETD(td_N, self.td_sizes, self.layer_shapes,
                                            self.layer_sizes))
        if rand_N > 0:  # Outdated
            self.decoder_methods.append(ERand(rand_N, self.layer_shapes,
                                              self.layer_sizes))
        if nn_N > 0:  # Outdated
            self.decoder_methods.append(ENN(nn_N, self.layer_shapes,
                                            self.layer_sizes))

        # Miscellaneous constants
        self.count = 0
        self.compress_len = self.compress_decoder(self.decoder()).size

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
        for e in self.decoder_methods:
            tret, decoder = e.expand_decoder(decoder)
            for key in tret:
                ret[key] = tret[key]

        return ret

    def decode(self, decoder):
        decoder = self.expand_decoder(decoder)

        ret = {}
        for i in range(1, len(self.layers)):
            ret['W' + str(i)] = np.zeros(shape=self.layer_shapes[i][0])
            ret['b' + str(i)] = np.zeros(shape=self.layer_shapes[i][1])
            ret['func' + str(i)] = self.layers[i][-1]

        for e in self.decoder_methods:
            tret = e.decode(decoder)
            for i in range(1, len(self.layers)):
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

    def set_tf_weights(self, model, weights):
        for i in range(0, len(self.layers)):
            if self.layers[i][0] == 'input' or \
                    self.layers[i][0] == 'pool' or \
                    self.layers[i][0] == 'flatten':
                continue
            model.layers[i - 1].set_weights([weights['W' + str(i)], weights['b' + str(i)]])

        return model

    def clip(self, decoder):
        ret = np.array([decoder[0]], copy=True)
        for e in self.decoder_methods:
            tret, decoder = e.clip(decoder)
            ret = np.concatenate((ret, tret))

        return ret

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

    def mut(self, decoder):
        ret = np.array([decoder[0]], copy=True)
        for e in self.decoder_methods:
            tret, decoder = e.mut(decoder[1:], self.mut_prob)
            ret = np.concatenate((ret, tret))
        return ret

    def run(self):
        # Preparing records and files
        history = open(self.folder_name + '/hist.txt', 'w+')
        foo = input('Load previous checkpoint? (y/n) ')
        if foo == 'y':
            load_name = input('Save name: ')
            start_iter = int(input('Iteration: '))
            pop = np.load('saves/egaSave-' + load_name + '/iter-' + str(start_iter) + '.npy')
            if self.problem_name[0:5] == 'Snake':
                food_arr = np.load('saves/egaSave-' + load_name + '/food.npy').tolist()
        else:
            start_iter = 1
            pop = np.array([np.zeros(self.compress_len)] * self.pop_size)
            for p in range(self.pop_size):
                pop[p] = self.compress_decoder(self.decoder())
                pop[p] = self.clip(pop[p])
            if self.problem_name[0:5] == 'Snake':
                randx = np.random.randint(1, 20, 3000)
                randy = np.random.randint(1, 20, 3000)
                food_arr = [[i, j] for i, j in zip(randx, randy)]  # Test with preset food positions
                np.save(self.folder_name + '/food.npy', food_arr)

        model = self.get_tf_model()
        for t in range(start_iter, self.iterations + 1):
            start = time.time()
            fitness = np.zeros(self.pop_size)
            if self.problem_name == 'MNIST':
                cce = np.zeros(self.pop_size)
                acc = np.zeros(self.pop_size)
            if self.problem_name[0:5] == 'Snake':
                steps = np.zeros(self.pop_size)
                score = np.zeros(self.pop_size)
                art_score = np.zeros(self.pop_size)
                move_distributions = np.zeros((self.pop_size, 4))

            for p in range(self.pop_size):
                model = self.set_tf_weights(model, self.decode(pop[p]))
                if self.problem_name == 'MNIST':
                    cce[p], acc[p] = self.problem.test(model)
                    fitness[p] = cce[p]
                if self.problem_name[0:5] == 'Snake':
                    steps[p], score[p], art_score[p], move_distributions[p], _, _, _ \
                        = self.problem.test(model, food_arr=food_arr)
                    fitness[p] = art_score[p]
            sort = np.argsort(-fitness)
            pop = pop[sort]
            fitness = fitness[sort]
            if self.problem_name == 'MNIST':
                cce = cce[sort]
                acc = acc[sort]
            if self.problem_name[0:5] == 'Snake':
                steps = steps[sort]
                score = score[sort]
                art_score = art_score[sort]
                move_distributions = move_distributions[sort]

            if self.problem_name == 'MNIST':
                avg_cce = 1.0 * np.sum(cce) / self.pop_size
                avg_acc = 1.0 * np.sum(acc) / self.pop_size
                history.write(str(t).zfill(6) + '     ' +
                              str('{:.2f}'.format(cce[0])).zfill(3) + '   ' +
                              str('{:.2f}'.format(acc[0])).zfill(5) + '   ' +
                              str('{:.2f}'.format(avg_cce)).zfill(3) + '   ' +
                              str('{:.2f}'.format(avg_acc)).zfill(5) + '\n')
            if self.problem_name[0:5] == 'Snake':
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

                if self.problem_name == 'MNIST':
                    print(str(t).zfill(6) + '     ' +
                          str('{:.2f}'.format(cce[0])).zfill(3) + '   ' +
                          str('{:.2f}'.format(acc[0])).zfill(5) + '   ' +
                          str('{:.2f}'.format(end - start).zfill(6)) + 's')
                if self.problem_name[0:5] == 'Snake':
                    print('Iter ' + str(t).zfill(6) + '   ' +
                          str(int(pop[0][0])).zfill(6) + '   ' +
                          str('{:.2f}'.format(steps[0])).zfill(7) + '   ' +
                          str('{:.2f}'.format(score[0])).zfill(6) + '   ' +
                          str('{:.2f}'.format(art_score[0])).zfill(7) + '     ' +
                          str(move_distributions[0][0]).zfill(5) + '   ' +
                          str(move_distributions[0][1]).zfill(5) + '   ' +
                          str(move_distributions[0][2]).zfill(5) + '   ' +
                          str(move_distributions[0][3]).zfill(5) + '          ' +
                          str('{:.2f}'.format(end - start).zfill(6)) + 's')

            prob = np.array(fitness, copy=True) - min(0, np.amin(fitness))  # prevent negatives
            prob = prob / np.sum(prob)
            if self.problem_name == 'MNIST':
                prob = 1 - prob  # Smaller is better
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
        if self.problem_name == 'MNIST':
            cce, acc = self.problem.test(model)
            print('Results:   ' +
                  str('{:.2f}'.format(cce[0])).zfill(3) + '   ' +
                  str('{:.2f}'.format(acc[0])).zfill(5))
        if self.problem_name[0:5] == 'Snake':
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
        print('==================================================')
        if self.problem_name[0:5] == 'Snake':
            input()
            self.problem.test(model, goal_steps=30, gui=True)
