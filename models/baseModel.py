from snakeGame import SnakeGame
from random import randint
import numpy as np
from collections import Counter


class Base:
    def __init__(self, initial_games=100, test_games=10, goal_steps=100, lr=1e-2):
        self.initial_games = initial_games
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.lr = lr
        self.vecs_to_keys = {(-1, 0): 0,
                             (0, 1): 1,
                             (1, 0): 2,
                             (0, -1): 3}

        self.training_data = self.initial_population()
        self.X = np.array([i[0] for i in self.training_data]).T
        self.y = np.array([i[1] for i in self.training_data]).T

    def initial_population(self):
        training_data = []
        for _ in range(self.initial_games):
            game = SnakeGame()
            _, _, snake, _ = game.start()
            prev_obs = self.gen_obs(snake)
            for _ in range(self.goal_steps):
                action, game_action = self.gen_action(snake)
                done, _, snake, _ = game.step(game_action)
                action_obs = np.append([action], prev_obs)
                if done:
                    training_data.append([action_obs, -1])
                    break
                else:
                    training_data.append([action_obs, 1])
                    prev_obs = self.gen_obs(snake)
        return training_data

    def gen_obs(self, snake):
        snake_dir = self.get_snake_dir_vec(snake)
        barr_left = self.is_dir_blocked(snake, self.turn_vec_left(snake_dir))
        barr_front = self.is_dir_blocked(snake, snake_dir)
        barr_right = self.is_dir_blocked(snake, self.turn_vec_right(snake_dir))
        return np.array([int(barr_left), int(barr_front), int(barr_right)])

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

    def turn_vec_left(self, vec):
        return np.array([-vec[1], vec[0]])

    def turn_vec_right(self, vec):
        return np.array([vec[1], -vec[0]])

    def is_dir_blocked(self, snake, snake_dir):
        point = np.array(snake[0]) + np.array(snake_dir)
        return point.tolist() in snake[:-1] or point[0] == 0 or point[1] == 0 or point[0] == 21 or point[1] == 21

    def model(self):
        W1 = np.random.randn(1, 4) * 0.01
        b1 = np.zeros(shape=(1, 1))
        return {'W1': W1, 'b1': b1}

    def forward_prop(self, model, X, y=None):
        W1 = model['W1']
        b1 = model['b1']

        Z1 = np.dot(W1, X) + b1
        A1 = Z1
        if y is None:
            J = -1
        else:
            J = 1 / 2 * np.mean((A1 - y) ** 2)
        return J, Z1, A1

    def test(self, model):
        steps_arr = []
        for _ in range(self.test_games):
            steps = 0
            game_mem = []
            game = SnakeGame()
            _, _, snake, _ = game.start()
            prev_obs = self.gen_obs(snake)
            for _ in range(self.goal_steps):
                preds = []
                for action in range(-1, 2):
                    preds.append(self.forward_prop(model, np.append([action], prev_obs))[-1])
                action = np.argmax(np.array(preds))
                game_action = self.get_game_action(snake, action - 1)
                done, _, snake, _ = game.step(game_action)
                game_mem.append([prev_obs, action])
                if done:
                    break
                else:
                    prev_obs = self.gen_obs(snake)
                    steps += 1
            steps_arr.append(steps)
        print('Average steps: ' + str(1.0 * sum(steps_arr) / len(steps_arr)))
        print(Counter(steps_arr))

    def visualize(self, model):
        game = SnakeGame(gui=True)
        _, _, snake, food = game.start()
        prev_obs = self.gen_obs(snake, food)
        for _ in range(self.goal_steps):
            preds = []
            for action in range(-1, 2):
                preds.append(self.forward_prop(model, np.append([action], prev_obs))[-1])
            action = np.argmax(np.array(preds))
            game_action = self.get_game_action(snake, action - 1)
            done, _, snake, food = game.step(game_action)
            if done:
                break
            else:
                prev_obs = self.gen_obs(snake, food)
        game.end_game()

    def run(self):
        model = self.model()
        m = self.X.shape[1]
        for _ in range(100):
            J, Z1, A1 = self.forward_prop(model, self.X, self.y)

            dZ1 = (A1 - self.y)
            dW1 = (1 / m) * np.dot(dZ1, self.X.T)
            db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

            model['W1'] -= self.lr * dW1
            model['b1'] -= self.lr * db1
        np.savez('../saves/baseSave.npz', W1=model['W1'], b1=model['b1'])
        self.test_model(model)

    def load(self):
        nn = self.model()
        npz = np.load('../saves/baseModel.npz')
        nn['W1'] = npz['W1']
        nn['b1'] = npz['b1']
        W1 = npz['W1']
        b1 = npz['b1']
        print(np.amax(W1))
        print(np.amin(W1))
        print(np.amax(b1))
        print(np.amin(b1))
        return nn


if __name__ == '__main__':
    base = Base()
    base.run()
