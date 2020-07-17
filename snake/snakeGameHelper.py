import numpy as np
from random import randint
import math
import neat
import tensorflow as tf


class Helper:
    def __init__(self):
        self.vecs_to_keys = {(-1, 0): 0,
                             (0, 1): 1,
                             (1, 0): 2,
                             (0, -1): 3}

    def gen_snake_obs(self, snake, width, height):
        board = np.zeros((width, height))
        for i in snake:
            board[i[0] - 1, i[1] - 1] = 1
        # board[food[0] - 1, food[1] - 1] = 1

        ret = []
        k = 1 << np.arange(width, dtype=np.uint32)[::-1]
        for i in range(height):
            ret.append(board[:, i].dot(k))
            # if i == food[0] - 1:
            #     ret[-1] *= (1 << 10)
        k = 1 << np.arange(height, dtype=np.uint32)[::-1]
        for i in range(width):
            ret.append(board[i, :].dot(k))
            # if i == food[1] - 1:
            #     ret[-1] *= (1 << 10)

        ret = np.array(ret)
        return ret

    def gen_food_obs(self, food, width, height):
        board = np.zeros((width, height))
        board[food[0] - 1][food[1] - 1] = 1

        ret = []
        k = 1 << np.arange(width, dtype=np.uint32)[::-1]
        for i in range(height):
            ret.append(board[:, i].dot(k))
            # if i == food[0] - 1:
            #     ret[-1] *= (1 << 10)
        k = 1 << np.arange(height, dtype=np.uint32)[::-1]
        for i in range(width):
            ret.append(board[i, :].dot(k))
            # if i == food[1] - 1:
            #     ret[-1] *= (1 << 10)

        ret = np.array(ret)
        return ret

        # snake_dir = self.get_snake_dir_vec(snake)
        # food_dir = self.get_food_dir_vec(snake, food)
        # barr_left = self.is_dir_blocked(snake, self.turn_vec_left(snake_dir))
        # barr_front = self.is_dir_blocked(snake, snake_dir)
        # barr_right = self.is_dir_blocked(snake, self.turn_vec_right(snake_dir))
        # angle = self.get_angle(snake_dir, food_dir)
        # return np.array([int(barr_left), int(barr_front), int(barr_right), angle])

    def gen_action(self, snake):
        action = randint(0, 2) - 1
        return action, self.get_game_action(snake, action)

    def get_game_action(self, snake, action):
        new_dir = self.get_snake_dir_vec(snake)
        if action == -1:
            new_dir = self.turn_vec_left(new_dir)
        if action == 1:
            new_dir = self.turn_vec_right(new_dir)
        game_action = self.vecs_to_keys[tuple(new_dir.tolist())]
        return game_action

    def get_snake_dir_vec(self, snake):
        return np.array(snake[0]) - np.array(snake[1])

    def get_food_dir_vec(self, snake, food):
        return np.array(food) - np.array(snake[0])

    def norm_vec(self, vec):
        return vec / np.linalg.norm(vec)

    def get_food_dist(self, snake, food):
        return np.linalg.norm(self.get_food_dir_vec(snake, food))

    def get_angle(self, a, b):
        a = self.norm_vec(a)
        b = self.norm_vec(b)
        return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]) / math.pi

    def turn_vec_left(self, vec):
        return np.array([-vec[1], vec[0]])

    def turn_vec_right(self, vec):
        return np.array([vec[1], -vec[0]])

    def is_dir_blocked(self, snake, snake_dir):
        point = np.array(snake[0]) + np.array(snake_dir)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def forward_prop(self, model, X, y=None):
        if type(model) is dict:
            W1 = model['W1']
            b1 = model['b1']
            W2 = model['W2']
            b2 = model['b2']

            Z1 = np.dot(W1, X) + b1
            A1 = np.maximum(Z1, 0)
            Z2 = np.dot(W2, A1) + b2
            A2 = Z2
            if y is None:
                J = -1
            else:
                J = 1 / 2 * np.mean((A2 - y) ** 2)

            return J, Z1, A1, Z2, A2

        if type(model) is neat.nn.feed_forward.FeedForwardNetwork:
            J = 0
            A2 = []
            if X.ndim == 1:
                X = np.expand_dims(X, axis=-1)
            for i in range(X.shape[1]):
                output = model.activate(X[:, i])
                A2.append(output)
                if y is None:
                    pass
                else:
                    J += (output - y[i]) ** 2
            if y is None:
                J = -1
            else:
                J = 1 / (2 * X.shape[1])
            A2 = np.array(A2)

            return J, A2

        if type(model) is tf.keras.Model:
            X = np.expand_dims(X, axis=0)
            A2 = model.predict(X)
            if y is None:
                J = -1
            else:
                J = 1 / 2 * np.mean((A2 - y) ** 2)

            return J, A2


if __name__ == '__main__':
    helper = Helper()
    snake = np.array([[1, 1], [2, 1], [3, 1], [3, 2]])
    print(helper.gen_obs(snake, [1, 1], 20, 20))
