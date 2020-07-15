
import sys
sys.path.insert(0, '../problems')

# from scoreProblem import ScoreProb
# from actionProblem import ActionProb
# from simulationProblem import SimulationProb

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
    def __init__(self, problem, initial_games=10000, test_games=10, goal_steps=2000, lr=1e-2, **kwargs):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.vecs_to_keys = {(-1, 0): 0,
                             (0, 1): 1,
                             (1, 0): 2,
                             (0, -1): 3}

        # if problem == 'Score':
        #     self.problem = ScoreProb()
        # if problem == 'Action':
        #     self.problem = ActionProb()
        # if problem == 'Simulation':
        #     self.problem = SimulationProb(kwargs['p'], kwargs['c'])
        # self.problem = None
        self.ln1, self.ln2, self.ln3 = self.problem.model_dims()

        self.clip_lo = -1
        self.clip_hi = 1

    def model(self):
        xIn = tf.keras.Input(shape=(self.ln1))
        x = tf.keras.layers.Dense(self.ln2, activation='relu',
                                  kernel_constraint=MinMax(self.clip_lo, self.clip_hi),
                                  bias_constraint=MinMax(self.clip_lo, self.clip_hi))(xIn)
        x = tf.keras.layers.Dense(self.ln3, activation='linear',
                                  kernel_constraint=MinMax(self.clip_lo, self.clip_hi),
                                  bias_constraint=MinMax(self.clip_lo, self.clip_hi))(x)
        model = tf.keras.Model(inputs=xIn, outputs=[x])
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=[])
        return model

    def save_npy(self):
        nn = self.load()
        layers = []
        for i, layer in enumerate(nn.layers[1:]):
            layers.append(np.array(layer.get_weights()))
        layers[0][0] = np.swapaxes(layers[0][0], 0, 1)
        print(layers[0][0].shape)
        layers[1][0] = np.swapaxes(layers[1][0], 0, 1)
        print(layers[1][0].shape)
        layers[0][1] = np.expand_dims(layers[0][1], axis=-1)
        print(layers[0][1].shape)
        layers[1][1] = np.expand_dims(layers[1][1], axis=-1)
        print(layers[1][1].shape)
        np.savez('../saves/bpSave.npz', W1=layers[0][0], b1=layers[0][1], W2=layers[1][0], b2=layers[1][1])

    def run(self):
        model = self.model()
        model.fit(self.problem.X.T, self.problem.y, epochs=3, shuffle=True)
        model.save('../saves/bpSave.h5')
        steps, score = self.problem.test(model, 10)
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6))
        print('==================================================')

    def test(self):
        model = self.model()
        model.load_weights('../saves/bpSave.h5')
        steps, score = self.problem.test(model, 10)
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6))
        print('==================================================')


if __name__ == '__main__':
    bp = BP('Score')
    bp.test()
