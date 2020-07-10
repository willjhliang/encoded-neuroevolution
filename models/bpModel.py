from snakeGame import SnakeGame
from random import randint
import numpy as np
import math
from collections import Counter


class BP:
    def __init__(self, initial_games=1000, test_games=1000, goal_steps=2000, lr=1e-2, file='bpSave.npz'):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.vecs_to_keys = {(-1, 0): 0,
                             (0, 1): 1,
                             (1, 0): 2,
                             (0, -1): 3}
        self.file = file

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
        print("Finished generating training data")
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
        W1 = np.random.randn(25, 5) * 0.01
        b1 = np.zeros(shape=(25, 1))
        W2 = np.random.randn(1, 25) * 0.01
        b2 = np.zeros(shape=(1, 1))
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def relu_der(self, dA, Z):
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        dZ[Z > 0] = 1
        return dZ

    def forward_prop(self, X, y, model):
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']

        Z1 = np.dot(W1, X) + b1
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(W2, A1) + b2
        A2 = Z2
        J = 1 / 2 * np.mean((A2 - y) ** 2)

        return J, Z1, A1, Z2, A2

    def train_model(self, training_data, model):
        X = np.array([i[0] for i in training_data]).T
        y = np.array([i[1] for i in training_data]).T
        m = X.shape[1]

        for _ in range(100):
            J, Z1, A1, Z2, A2 = self.forward_prop(X, y, model)

            dZ2 = (A2 - y)
            dW2 = (1 / m) * np.dot(dZ2, A1.T)
            db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
            W2 = model['W2']
            dZ1 = self.relu_der(np.dot(W2.T, dZ2), Z1)
            dW1 = (1 / m) * np.dot(dZ1, X.T)
            db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

            model['W1'] -= self.lr * dW1
            model['b1'] -= self.lr * db1
            model['W2'] -= self.lr * dW2
            model['b2'] -= self.lr * db2
        np.savez('saves/' + self.file, W1=model['W1'], b1=model['b1'], W2=model['W2'], b2=model['b2'])
        return model

    def predict(self, X, model):
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']

        X = np.expand_dims(X, axis=-1)

        Z1 = np.dot(W1, X) + b1
        A1 = np.maximum(Z1, 0)
        Z2 = np.dot(W2, A1) + b2
        A2 = Z2
        return A2

    def test_model(self, model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_mem = []
            game = SnakeGame()
            _, score, snake, food = game.start()
            prev_obs = self.gen_obs(snake, food)
            for _ in range(self.goal_steps):
                preds = []
                for action in range(-1, 2):
                    preds.append(self.predict(np.append([action], prev_obs), model))
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

    def test_neat_model(self, model):
        steps_arr = []
        scores_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_mem = []
            game = SnakeGame()
            _, score, snake, food = game.start()
            prev_obs = self.gen_obs(snake, food)
            for _ in range(self.goal_steps):
                preds = []
                for action in range(-1, 2):
                    preds.append(model.activate(np.append([action], prev_obs)))
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

    def visualize_game(self, model):
        game = SnakeGame(gui=True)
        _, _, snake, food = game.start()
        prev_obs = self.gen_obs(snake, food)
        for _ in range(self.goal_steps):
            preds = []
            for action in range(-1, 2):
                preds.append(self.predict(np.append([action], prev_obs), model))
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

    def load(self, file=None):
        if file is None:
            file = self.file
        nn = self.model()
        npz = np.load('../saves/' + file)
        nn['W1'] = npz['W1']
        nn['b1'] = npz['b1']
        nn['W2'] = npz['W2']
        nn['b2'] = npz['b2']
        return nn

    def visualize(self):
        nn = self.load('../saves/tfSave.npz')
        self.visualize_game(nn)

    def test(self):
        nn = self.load('../saves/tfSave.npz')
        self.test_model(nn)


if __name__ == '__main__':
    BP = BP()
    BP.train()
