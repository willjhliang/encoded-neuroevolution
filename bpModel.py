
import sys
sys.path.insert(0, '../problems')

from scoreProblem import ScoreProb
from actionProblem import ActionProb

import numpy as np
import os

import tensorflow as tf


class MinMax(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value):
        self.min_val = min_value
        self.max_val = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_val, self.max_val)


class BP:
    def __init__(self, problem, lr=1e-2):
        self.lr = lr
        self.vecs_to_keys = {(-1, 0): 0,
                             (0, 1): 1,
                             (1, 0): 2,
                             (0, -1): 3}

        self.problem = None
        if problem == 'Score':
            self.problem = ScoreProb()
        if problem == 'Action':
            self.problem = ActionProb()

        self.ln = [[800, 1], [800, 24], [24, 16], [16, 8], [8, 4]]
        self.lfunc = ['relu', 'relu', 'relu', 'linear']

    def model(self):
        xIn = tf.keras.Input(shape=(5))
        x = tf.keras.layers.Dense(25, activation='relu')(xIn)
        x = tf.keras.layers.Dense(1, activation='linear')(x)
        # xIn = tf.keras.Input(shape=(self.ln[0][0]))
        # x = tf.keras.layers.Dense(self.ln[1][-1], activation=self.lfunc[0])(xIn)
        # for i in range(2, len(self.ln)):
        #     x = tf.keras.layers.Dense(self.ln[i][-1], activation=self.lfunc[i-1])(x)
        model = tf.keras.Model(inputs=xIn, outputs=[x])
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=[])

#         ret = tf.keras.Sequential()
#         ret.add(tf.keras.layers.InputLayer(input_shape=(20, 20, 3)))
#         ret.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation=self.lfunc[0], padding='same'))
#         ret.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
#         ret.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5, activation=self.lfunc[1], padding='same'))
#         ret.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
#         ret.add(tf.keras.layers.Flatten())
#         ret.add(tf.keras.layers.Dense(64, activation=self.lfunc[2]))
#         ret.add(tf.keras.layers.Dense(4, activation=self.lfunc[3]))
#         ret.compile(optimizer='adam',
#                     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
#                     metrics=['mse'])
        return model

    def save_npy(self):
        model = self.model()
        model.load_weights('saves/bpSave.h5')
        layers = []
        for i, layer in enumerate(model.layers[1:]):
            layers.append(np.array(layer.get_weights()))
        np.save('saves/bpSave.npy', layers)

    def run(self):
        data = self.problem.get_training_data()
        model = self.model()
        # m = data['X'].shape[0]
        # X = data['X'].reshape(m, 20, 20, 3)
        # y = data['y']
        # model.fit(x=X, y=y[:, :, 0], epochs=25, shuffle=True)
        model.fit(x=data['X'], y=data['y'], epochs=50, shuffle=True)
        model.save('saves/bpScoreSave.h5')
        fitness, steps, score = self.problem.test(model, 30)
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6))
        print('==================================================')

    def test(self):
        model = self.model()
        model.load_weights('saves/bpScoreSave.h5')
        fitness, steps, score = self.problem.test(model, 30)
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6))
        print('==================================================')
