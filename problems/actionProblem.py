
import sys
sys.path.insert(0, '../snake')

from snakeGame import SnakeGame
from snakeGameHelper import Helper

import os
import numpy as np


class ActionProb:
    def __init__(self, p, c, test_games=10, goal_steps=2000):
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.helper = Helper()

        self.p = p
        self.c = c

        # self.training_data = self.initial_population()
        # self.X = np.array([i[0] for i in self.training_data]).T
        # self.y = np.array([i[1] for i in self.training_data]).T

    def model(self):
        W1 = np.random.randn(20, 4) * 0.01
        b1 = np.zeros(shape=(20, 1))
        W2 = np.random.randn(3, 20) * 0.01
        b2 = np.zeros(shape=(3, 1))
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def model_dims(self):
        return 4, 20, 3

    def test(self, model, games=None, gui=False):
        if games is None:
            games = self.test_games
            # return self.helper.forward_prop(model, self.X, self.y)[0]

        steps_arr = []
        scores_arr = []
        for _ in range(games):
            steps = 0
            game_mem = []
            game = SnakeGame(gui=gui)
            _, score, snake, food = game.start()
            prev_obs = self.helper.gen_obs(snake, food)
            for _ in range(self.goal_steps):
                preds = self.helper.forward_prop(model, prev_obs)[-1]
                action = np.argmax(np.array(preds))
                game_action = self.helper.get_game_action(snake, action - 1)
                done, score, snake, food = game.step(game_action)
                game_mem.append([prev_obs, action])
                if done:
                    break
                else:
                    prev_obs = self.helper.gen_obs(snake, food)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
        game.end_game()

        avg_steps = 1.0 * sum(steps_arr) / len(steps_arr)
        avg_score = 1.0 * sum(scores_arr) / len(scores_arr)
        fitness = avg_score - self.p * max(0, self.c * self.goal_steps - avg_steps)
        return fitness, avg_steps, avg_score

    # def initial_population(self):
    #     if os.path.isfile('../data/actionData.npy'):
    #         return np.load('../data/actionData.npy', allow_pickle=True)
    #         print('Loaded observation -> action data')

    #     training_data = []
    #     bpModel = BP('Score')
    #     bpModel.run()
    #     model = bpModel.model
    #     for i in range(self.initial_games):
    #         game = SnakeGame()
    #         _, prev_score, snake, food = game.start()
    #         prev_obs = self.helper.gen_obs(snake, food)
    #         for _ in range(self.goal_steps):
    #             preds = []
    #             for action in range(-1, 2):
    #                 preds.append(model.predict(np.append([action], prev_obs).reshape(5, -1))[-1])
    #             action = np.argmax(np.array(preds))
    #             game_action = self.get_game_action(snake, action - 1)
    #             done, score, snake, food = game.step(game_action)
    #             vec = np.zeros((3, 1))
    #             vec[action] = 1
    #             if done:
    #                 break
    #             else:
    #                 training_data.append([prev_obs, vec])
    #                 prev_obs = self.helper.gen_obs(snake, food)
    #     print('Generated observation -> action data')
    #     np.save('../data/actionData.npy', np.array(training_data))
    #     return training_data
