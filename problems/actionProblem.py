
import sys
sys.path.insert(0, '../snake')

from snakeGame import SnakeGame
from snakeGameHelper import Helper

import os
import numpy as np


class ActionProb:
    def __init__(self, p, c, test_games=20, goal_steps=2000, num_frames=3):
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.num_frames = num_frames
        game = SnakeGame()
        self.helper = Helper(game.board['width'], game.board['height'], 3)

        self.p = p
        self.c = c

        # self.training_data = self.initial_population()
        # self.X = np.array([i[0] for i in self.training_data]).T
        # self.y = np.array([i[1] for i in self.training_data]).T

    def model(self):
        return -1
        # ret = {}
        # for i in range(1, len(self.ln)):
        #     W = np.random.randn(self.ln[i], self.ln[i - 1]) * 0.01
        #     b = np.zeros(shape=(self.ln[i], 1))
        #     ret['W' + str(i)] = W
        #     ret['b' + str(i)] = b
        #     ret['func' + str(i)] = self.lfunc[i - 1]
        #     print(ret['func1'])
        # return ret
        # W1 = np.random.randn(40, self.num_frames * 40 + 40) * 0.01
        # # W1 = np.random.randn(20, 4) * 0.01
        # b1 = np.zeros(shape=(40, 1))
        # W2 = np.random.randn(3, 40) * 0.01
        # b2 = np.zeros(shape=(3, 1))
        # return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def test(self, model, games=None, goal_steps=None, gui=False):
        if games is None:
            games = self.test_games
            # return self.helper.forward_prop(model, self.X, self.y)[0]
        if goal_steps is None:
            goal_steps = self.goal_steps

        steps_arr = []
        scores_arr = []
        game_score_arr = []
        for _ in range(games):
            steps = 0
            score = 0
            game_mem = []
            game = SnakeGame(gui=gui)
            _, game_score, snake, food = game.start()
            obs = self.helper.gen_obs(snake, food, 'Pix')

            prev_dist = -1
            prev_game_score = 0
            for _ in range(goal_steps):
                preds = self.helper.forward_prop(model, obs)[-1]
                action = np.argmax(np.array(preds))
                game_action = self.helper.get_game_action(snake, action - 1)
                done, game_score, snake, food = game.step(game_action)
                game_mem.append([obs, action])

                dist = (abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1]))
                if game_score == prev_game_score + 1:
                    score += 50
                if dist < prev_dist:
                    score += 1
                prev_game_score = game_score
                prev_dist = dist

                if done:
                    break
                else:
                    obs = self.helper.update_obs(snake, food, 'Pix', obs)
                    steps += 1
            steps_arr.append(steps)
            scores_arr.append(score)
            game_score_arr.append(game_score)
        game.end_game()

        avg_steps = 1.0 * sum(steps_arr) / len(steps_arr)
        avg_score = 1.0 * sum(scores_arr) / len(scores_arr)
        avg_game_score = 1.0 * sum(game_score_arr) / len(game_score_arr)
        fitness = 10 * avg_score + avg_steps
        return fitness, avg_steps, avg_score, avg_game_score

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
