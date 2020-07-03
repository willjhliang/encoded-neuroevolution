from snakeGame import SnakeGame
from random import randint
import numpy as np
import math
from collections import Counter

import tensorflow as tf


class SnakeNN:
    def __init__(self, initial_games=10000, test_games=10, goal_steps=2000, lr=1e-1, filename='model.h5'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.vecs_to_keys = {(-1, 0): 0,
                             (0, 1): 1,
                             (1, 0): 2,
                             (0, -1): 3}
        self.filename = filename

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = SnakeGame()
            _, prev_score, snake, food = game.start()
            prev_obs = self.gen_obs(snake, food)
            prev_food_dist = self.get_food_dist(snake, food)
            for _ in range(self.goal_steps):
                action, game_action = self.gen_action(snake)
                done, score, snake, food = game.step(game_action)
                action_obs = np.append([action], prev_obs)
                if done:
                    training_data.append([action_obs, -1])
                    break
                else:
                    food_dist = self.get_food_dist(snake, food)
                    if score > prev_score or food_dist < prev_food_dist:
                        training_data.append([action_obs, 1])
                    else:
                        training_data.append([action_obs, 0])
                    prev_obs = self.gen_obs(snake, food)
                    prev_food_dist = food_dist
        return training_data

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
        xIn = tf.keras.Input(shape=(5))
        x = tf.keras.layers.Dense(25, activation='relu')(xIn)
        x = tf.keras.layers.Dense(1, activation='linear')(x)
        model = tf.keras.Model(inputs=xIn, outputs=[x])
        model.compile(optimizer='adam',
                      loss='mse',
                      metrics=[])
        return model

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).reshape(-1, 5)
        y = np.array([i[1] for i in training_data]).reshape(-1, 1)
        model.fit(X, y, epochs=3, shuffle=True)
        model.save(self.filename)
        return model

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for i in range(self.test_games):
            steps = 0
            game_mem = []
            game = SnakeGame()
            _, score, snake, food = game.start()
            prev_obs = self.gen_obs(snake, food)
            for _ in range(self.goal_steps):
                preds = []
                for action in range(-1, 2):
                    preds.append(model.predict(np.append([action], prev_obs).reshape(-1, 5)))
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
        print('Average steps: ' + str(1.0 * sum(steps_arr) / len(steps_arr)))
        print(Counter(steps_arr))
        print('Average score: ' + str(1.0 * sum(scores_arr) / len(scores_arr)))
        print(Counter(scores_arr))
        return 1.0 * sum(scores_arr) / len(scores_arr)

    def visualize_game(self, model):
        game = SnakeGame(gui=True)
        _, _, snake, food = game.start()
        prev_obs = self.gen_obs(snake, food)
        for _ in range(self.goal_steps):
            preds = []
            for action in range(-1, 2):
                preds.append(model.predict(np.append([action], prev_obs).reshape(-1, 5)))
            action = np.argmax(np.array(preds))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food = game.step(game_action)
            if done:
                break
            else:
                prev_obs = self.gen_obs(snake, food)
        game.end_game()

    def train(self):
        training_data = self.initial_population()
        nn = self.model()
        nn = self.train_model(training_data, nn)
        self.test_model(nn)

    def load(self):
        nn = tf.keras.models.load_model(self.filename)
        return nn

    def visualize(self):
        nn = self.load()
        self.visualize_game(nn)

    def test(self):
        nn = self.load()
        self.test_model(nn)

    def extract_weights(self):
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
        np.savez('model_tf.npz', W1=layers[0][0], b1=layers[0][1], W2=layers[1][0], b2=layers[1][1])


if __name__ == '__main__':
    snakeNN = SnakeNN()
    snakeNN.extract_weights()
