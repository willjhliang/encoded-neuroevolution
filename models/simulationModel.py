from snakeGame import SnakeGame
from scoreModel import ScoreModel
from random import randint
import numpy as np
import math
from collections import Counter


class SimulationModel:
    def __init__(self, initial_games=100, test_games=1000, goal_steps=2000, lr=1e-2, file='actionSave.npz'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.vecs_to_keys = {(-1, 0): 0,
                             (0, 1): 1,
                             (1, 0): 2,
                             (0, -1): 3}
        self.file = file

        self.p = .1
        self.c = 0.9

    def gen_obs(self, snake, food):
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

    def model(self):
        W1 = np.random.randn(20, 4) * 0.01
        b1 = np.zeros(shape=(20, 1))
        W2 = np.random.randn(3, 20) * 0.01
        b2 = np.zeros(shape=(3, 1))
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def model_dims(self):
        return 4, 20, 3

    def relu_der(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dZ[Z > 0] = 1
        return dZ

    def forward_prop(self, model, X, y=None):
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

    def test(self, model, games=None):
        if games is None:
            games = self.test_games
        steps_arr = []
        scores_arr = []
        for _ in range(games):
            steps = 0
            game_mem = []
            game = SnakeGame()
            _, score, snake, food = game.start()
            prev_obs = self.gen_obs(snake, food)
            for _ in range(self.goal_steps):
                if type(model) is dict:
                    preds = self.forward_prop(model, prev_obs)[-1]
                else:
                    preds = model.activate(prev_obs)
                action = np.argmax(np.array(preds))
                game_action = self.get_game_action(snake, action - 1)
                done, score, snake, food = game.step(game_action)
                game_mem.append([prev_obs, action])
                if done:
                    break
                else:
                    prev_obs = self.gen_obs(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        # print('Average steps: ' + str(1.0 * sum(steps_arr) / len(steps_arr)))
        # print(Counter(steps_arr))
        # print('Average score: ' + str(1.0 * sum(scores_arr) / len(scores_arr)))
        # print(Counter(scores_arr))
        avg_steps = 1.0 * sum(steps_arr) / len(steps_arr)
        avg_score = 1.0 * sum(scores_arr) / len(scores_arr)
        return (avg_score - self.p * max(0, self.c * self.goal_steps - avg_steps)), avg_steps, avg_score
