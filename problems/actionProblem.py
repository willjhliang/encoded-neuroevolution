
import sys
sys.path.insert(0, '../snake')

from snakeGame import SnakeGame
from snakeGameHelper import Helper

import os
import numpy as np
import tensorflow as tf


class ActionProb:
    def __init__(self, test_games=1, goal_steps=2000, test_steps=8000, num_frames=3):
        self.test_games = test_games
        self.goal_steps = goal_steps
        self.test_steps = test_steps
        self.num_frames = num_frames
        game = SnakeGame(20, 20)
        self.helper = Helper(game.board['width'], game.board['height'], 3, '2Pix')

    def test_over_games(self, model, test_games=None, goal_steps=None, food_arr=None, gui=False):
        if test_games is None:
            test_games = self.test_games
        if goal_steps is None:
            goal_steps = self.goal_steps

        score_arr = []
        steps_arr = []
        art_score_arr = []
        move_distributions = np.array([0, 0, 0, 0])
        for g in range(test_games):
            game_mem = []
            game = SnakeGame(food_arr=food_arr, gui=gui)
            _, cur_score, snake, food = game.start()
            obs = self.helper.gen_obs(snake, food)
            steps = 0
            art_score = 0

            prev_dist = (abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1]))
            prev_score = 0
            prev_action = 1
            for s in range(goal_steps):
                preds = self.helper.forward_prop(model, obs)[-1]
                action = np.argmax(preds)
                if preds.size == 3:  # Relative direction
                    action = self.helper.get_game_action(snake, action - 1)
                if preds.size == 4:  # Absolute direction
                    if abs(action - prev_action) == 2:  # Went backwards
                        preds[action] = np.min(preds) - 1
                        action = np.argmax(preds)
                done, cur_score, snake, food = game.step(action)
                move_distributions[action] += 1
                game_mem.append([obs, action])

                steps += 1
                dist = (abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1]))
                # if dist < prev_dist:
                #     art_score += 1.0 / (2 ** dist)
                # art_score += 1.0 / (2 ** dist)
                if cur_score == prev_score + 1:
                    art_score += 30
                    # art_score += 1000.0
                if dist < prev_dist:
                    art_score += 1
                else:
                    art_score -= 1
                prev_score = cur_score
                prev_dist = dist
                prev_action = action

                if done:
                    score_arr.append(cur_score)
                    steps_arr.append(steps)
                    art_score_arr.append(art_score)
                    break
                else:
                    obs = self.helper.update_obs(snake, food, obs)
        game.end_game()

        steps_arr = np.array(steps_arr)
        score_arr = np.array(score_arr)
        art_score_arr = np.array(art_score_arr)
        avg_steps = 1.0 * np.sum(steps_arr) / test_games
        avg_score = 1.0 * np.sum(score_arr) / test_games
        avg_art_score = 1.0 * np.sum(art_score_arr) / test_games
        return avg_steps, avg_score, avg_art_score, move_distributions, steps_arr, score_arr, art_score_arr

    # def test_over_steps(self, model, test_steps=None, food_arr=None, gui=False):
    #     if test_steps is None:
    #         test_steps = self.test_steps

    #     score = 0
    #     art_score = 0
    #     move_distributions = np.array([0, 0, 0, 0])
    #     games = 0

    #     dead = True
    #     for s in range(test_steps):
    #         if dead:
    #             game_mem = []
    #             game = SnakeGame(food_arr=food_arr, gui=gui)
    #             _, cur_score, snake, food = game.start()
    #             obs = self.helper.gen_obs(snake, food)
    #             prev_dist = (abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1]))
    #             prev_score = 0
    #             games += 1
    #             dead = False

    #         preds = self.helper.forward_prop(model, obs)[-1]
    #         action = np.argmax(np.array(preds))
    #         if preds.size == 3:  # Relative direction
    #             action = self.helper.get_game_action(snake, action - 1)
    #         if preds.size == 4:  # Absolute direction
    #             pass
    #         dead, cur_score, snake, food = game.step(action)
    #         move_distributions[action] += 1
    #         game_mem.append([obs, action])

    #         dist = (abs(snake[0][0] - food[0]) + abs(snake[0][1] - food[1]))
    #         if cur_score == prev_score + 1:
    #             art_score += 30
    #         if dist < prev_dist:
    #             # art_score += (1.0 / dist) + 1
    #             art_score += 1
    #         else:
    #             art_score -= 1

    #         prev_score = cur_score
    #         prev_dist = dist
    #         if not dead:
    #             obs = self.helper.update_obs(snake, food, obs)
    #     game.end_game()

    #     return games, score, art_score, move_distributions

    # def get_training_data(self, file):
    #     # if os.path.isfile('data/' + file + '.npz'):
    #     #     print('Loaded observation -> action data')
    #     #     return np.load('data/' + file + '.npz', allow_pickle=True)

    #     model = tf.keras.load_model('saves/bpScoreSave.h5')
    #     pix_abs_ret = {'X': [], 'y': []}
    #     vec_abs_ret = {'X': [], 'y': []}
    #     pix_rel_ret = {'X': [], 'y': []}
    #     vec_rel_ret = {'X': [], 'y': []}
    #     for i in range(150):
    #         game = SnakeGame()
    #         _, prev_score, snake, food = game.start()
    #         pix_obs = self.helper.gen_obs(snake, food, 'Pix')
    #         vec_obs = self.helper.gen_obs(snake, food, 'Vec')
    #         for _ in range(self.test_steps):
    #             preds = []
    #             for action in range(-1, 2):
    #                 preds.append(self.helper.forward_prop(model, np.append([action], vec_obs))[-1])
    #             action = np.argmax(np.array(preds))
    #             game_action = self.helper.get_game_action(snake, action - 1)
    #             done, score, snake, food = game.step(game_action)
    #             rel_vec = np.zeros((3, 1))
    #             rel_vec[action] = 1
    #             abs_vec = np.zeros((4, 1))
    #             abs_vec[game_action] = 1
    #             if done:
    #                 break
    #             else:
    #                 pix_abs_ret['X'].append(pix_obs)
    #                 pix_abs_ret['y'].append(abs_vec)
    #                 vec_abs_ret['X'].append(vec_obs)
    #                 vec_abs_ret['y'].append(abs_vec)
    #                 pix_rel_ret['X'].append(pix_obs)
    #                 pix_rel_ret['y'].append(rel_vec)
    #                 vec_rel_ret['X'].append(vec_obs)
    #                 vec_rel_ret['y'].append(rel_vec)
    #                 pix_obs = self.helper.update_obs(snake, food, 'Pix', pix_obs)
    #                 vec_obs = self.helper.update_obs(snake, food, 'Vec', vec_obs)
    #     print('Generated observation -> action data')
    #     np.savez('data/actionPixAbsData.npz', X=pix_abs_ret['X'], y=pix_abs_ret['y'])
    #     np.savez('data/actionVecAbsData.npz', X=pix_abs_ret['X'], y=pix_abs_ret['y'])
    #     np.savez('data/actionPixRelData.npz', X=pix_rel_ret['X'], y=pix_rel_ret['y'])
    #     np.savez('data/actionVecRelData.npz', X=pix_rel_ret['X'], y=pix_rel_ret['y'])
    #     # return np.load('data/' + file + '.npz', allow_picke=True)

    # def model(self):
    #     return -1
    #     ret = {}
    #     for i in range(1, len(self.ln)):
    #         W = np.random.randn(self.ln[i], self.ln[i - 1]) * 0.01
    #         b = np.zeros(shape=(self.ln[i], 1))
    #         ret['W' + str(i)] = W
    #         ret['b' + str(i)] = b
    #         ret['func' + str(i)] = self.lfunc[i - 1]
    #         print(ret['func1'])
    #     return ret
    #     W1 = np.random.randn(40, self.num_frames * 40 + 40) * 0.01
    #     # W1 = np.random.randn(20, 4) * 0.01
    #     b1 = np.zeros(shape=(40, 1))
    #     W2 = np.random.randn(3, 40) * 0.01
    #     b2 = np.zeros(shape=(3, 1))
    #     return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
