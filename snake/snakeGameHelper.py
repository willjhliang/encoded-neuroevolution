import numpy as np
from random import randint
import math
import neat
import tensorflow as tf


class Helper:
    def __init__(self, width, height, num_frames):
        self.vecs_to_keys = {(-1, 0): 0,
                             (0, 1): 1,
                             (1, 0): 2,
                             (0, -1): 3}
        self.width = width
        self.height = height
        self.num_frames = num_frames

    def gen_obs(self, snake, food, intype):
        if intype == 'Vec':
            return self.gen_vec_obs(snake, food)
        if intype == 'Pix':
            # ret = []
            # for i in range(self.num_frames):
            #     ret.append(self.gen_pix_snake_obs(snake))
            # ret.append(self.gen_pix_food_obs(food))
            # return np.array(ret)

            board1 = np.zeros((self.width, self.height))
            board2 = np.zeros((self.width, self.height))
            for i in snake:
                board1[i[0] - 1, i[1] - 1] = 1
            board2[food[0] - 1][food[1] - 1] = 1
            return np.stack((board1, board2), axis=2)
            # return np.concatenate((board1.flatten(), board2.flatten()))
            # return board1.flatten()

        return -1

    def update_obs(self, snake, food, intype, obs):
        if intype == 'Vec':
            return self.gen_vec_obs(snake, food)
        if intype == 'Pix':
            # ret = obs.tolist()
            # for i in range(self.num_frames - 1):
            #     ret[i + 1] = ret[i]
            # ret[0] = self.gen_pix_snake_obs(snake)
            # ret[-1] = self.gen_pix_food_obs(food)
            # return np.array(ret)

            board1 = np.zeros((self.width, self.height))
            board2 = np.zeros((self.width, self.height))
            for i in snake:
                board1[i[0] - 1, i[1] - 1] = 1
            board2[food[0] - 1][food[1] - 1] = 1
            return np.stack((board1, board2), axis=2)
            # return np.concatenate((board1.flatten(), board2.flatten()))
            # return board1.flatten()

        return -1

    # def get_board(self, snake, food, intype, **kwargs):
    #     board = np.zeros((kwargs['width'], kwargs['height']))
    #     for i in snake:
    #         board[i[0] - 1, i[1] - 1] = 1
    #     board[food[0] - 1][food[1] - 1] = -100
    #     return board

    def gen_pix_snake_obs(self, snake):
        board = np.zeros((self.width, self.height))
        for i in snake:
            board[i[0] - 1, i[1] - 1] = 1
        # board[food[0] - 1, food[1] - 1] = 1

        ret = []
        for i in range(0, self.height, 2):
            for j in range(0, self.width, 2):
                k = board[i][j] + board[i + 1][j] * 4 + board[i][j + 1] * 16 + board[i + 1][j + 1] * 64
                ret.append(k)
        return np.array(ret)

        # ret = []
        # k = 1 << np.arange(width, dtype=np.uint32)[::-1]
        # for i in range(height):
        #     ret.append(board[:, i].dot(k))
        #     # if i == food[0] - 1:
        #     #     ret[-1] *= (1 << 10)
        # k = 1 << np.arange(height, dtype=np.uint32)[::-1]
        # for i in range(width):
        #     ret.append(board[i, :].dot(k))
        #     # if i == food[1] - 1:
        #     #     ret[-1] *= (1 << 10)
        # ret = np.array(ret)
        # return ret

    def gen_pix_food_obs(self, food):
        board = np.zeros((self.width, self.height))
        board[food[0] - 1][food[1] - 1] = 1

        ret = []
        for i in range(0, self.height, 2):
            for j in range(0, self.width, 2):
                k = board[i][j] + board[i + 1][j] * 4 + board[i][j + 1] * 16 + board[i + 1][j + 1] * 64
                ret.append(k)
        return np.array(ret)

#         ret = []
#         k = 1 << np.arange(width, dtype=np.uint32)[::-1]
#         for i in range(height):
#             ret.append(board[:, i].dot(k))
#             # if i == food[0] - 1:
#             #     ret[-1] *= (1 << 10)
#         k = 1 << np.arange(height, dtype=np.uint32)[::-1]
#         for i in range(width):
#             ret.append(board[i, :].dot(k))
#             # if i == food[1] - 1:
#             #     ret[-1] *= (1 << 10)
#         ret = np.array(ret)
#         return ret

    def gen_vec_obs(self, snake, food):
        snake_dir = self.get_snake_dir_vec(snake)
        food_dir = self.get_food_dir_vec(snake, food)
        barr_left = self.is_dir_blocked(snake, self.turn_vec_left(snake_dir))
        barr_front = self.is_dir_blocked(snake, snake_dir)
        barr_right = self.is_dir_blocked(snake, self.turn_vec_right(snake_dir))
        angle = self.get_angle(snake_dir, food_dir)
        return np.array([int(barr_left), int(barr_front), int(barr_right), angle])

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
            A = np.array(X, copy=True)
            for i in range(1, int(len(model)/ 3)):
                W = model['W' + str(i)]
                b = model['b' + str(i)]
                func = model['func' + str(i)]

                Z = np.dot(W, A) + b
                if func == 'relu':
                    A = np.maximum(Z, 0)
                if func == 'sigmoid':
                    A = 1 / (1 + np.exp(-Z))
                if func == 'tanh':
                    A = np.tanh(Z)
                else:
                    A = Z

            if y is None:
                J = -1
            else:
                J = 1 / 2 * np.mean((A - y) ** 2)

            return J, A

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

        if type(model) is tf.keras.Sequential:
            X = np.expand_dims(X, axis=0)
            A = model.predict(X)
            if y is None:
                J = -1
            else:
                J = 1 / 2 * np.mean((A - y) ** 2)

            return J, A


if __name__ == '__main__':
    helper = Helper(20, 20, 1)
    snake = np.array([[1, 1], [2, 1], [3, 1], [3, 2]])
    print(helper.gen_obs(snake, [1, 1], 'Pix').shape)
