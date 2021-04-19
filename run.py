
import sys
sys.path.insert(0, 'encoders')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'problems/snake')

import os
import shutil
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
np.set_printoptions(suppress=True)  # Disables scientific notation
from egaModel import EGA
from psoModel import PSO

import cProfile
import pstats


def optimize_td_mut_scale(runs_per_pos=1):
    pos = [10 ** a for a in range(-10, -12, -1)]
    print(pos)
    res = []
    for scale in pos:
        print('------------------------------')
        print('SCALE: ' + str(scale))
        r = 0
        for i in range(runs_per_pos):
            ega = EGA('weights', 16, iterations=3000,
                      run_name=run_name, ckpt_period=100,
                      load_info=['False', '_', '_'], td_mut_scale_b=scale)
            r += ega.run()
        res.append(r / runs_per_pos)
    return res


def print_example():
    pop = np.load('saves/ega-weights-' + load_info[1] + '/iter-' +
                  load_info[2] + '.npy')
    ega = EGA('weights', 16, iterations=-1,
              run_name=run_name, ckpt_period=-1,
              load_info=load_info)
    res = ega.decode(pop[0])
    ega.test(pop)
    return res['W0']


def profile():
    profile = cProfile.Profile()
    profile.runcall(profileCall)
    ps = pstats.Stats(profile)
    ps.print_stats()


def profileCall():
    # Make sure to set problem as MINIMIZED weights
    ega = EGA('weights', 16, iterations=1000,
              run_name=run_name, ckpt_period=100,
              load_info=load_info)
    ega.run()


def testError():
    ega = EGA('weights', 16, iterations=1, pop_size=2,
              run_name=run_name, ckpt_period=1,
              load_info=load_info)
    ind = ega.compress_decoder(ega.get_decoder(custom_data=True))
    error = ega.problem.test(ega.decode(ind)['W0'])
    return error


def stepRank(rank):
    past_pop = np.array([])
    load_iter = 0
    iterations_per = 1000

    # td_mut_scale_V = 2e-1
    # td_mut_scale_a = 2e-3
    # td_mut_scale_b = 2e-5
    for r in range(1, rank + 1):
        ega = EGA('weights', r, iterations=iterations_per, pop_size=200,
                  run_name=run_name, ckpt_period=100,
                  load_info=['False', '_', str(load_iter)], plateau_len=200)
        ega.initialize_pop()

        # Set up population with previous run
        if r > 1:
            pop = np.array([np.zeros(ega.compress_len)] * ega.pop_size)
            uncompressed_pop = []
            for p in range(ega.pop_size):
                uncompressed_pop.append(ega.get_decoder())
            for p in range(ega.pop_size):
                for key in past_pop[p]:
                    if key == 'id':
                        uncompressed_pop[p][key] = past_pop[p][key]
                    elif key[1] == 'V':
                        for i, v in enumerate(past_pop[p][key]):
                            uncompressed_pop[p][key][i][:-1, ...] = v
                    elif key[1] == 'a':
                        uncompressed_pop[p][key][:-1] = past_pop[p][key]
                pop[p] = ega.compress_decoder(uncompressed_pop[p])
                pop[p] = ega.clip(pop[p])

            ega.pop = pop

        ega.run()

        load_iter += iterations_per
        past_pop = []
        for p in range(ega.pop_size):
            past_pop.append(ega.expand_decoder(ega.pop[p]))

        # td_mut_scale_V = max(1e-2, td_mut_scale_V - 1e-2)
        # td_mut_scale_a = max(1e-4, td_mut_scale_a - 1e-4)
        # td_mut_scale_b = max(1e-6, td_mut_scale_b - 1e-6)


def overwrite_save(algo):
    if os.path.isdir(algo.run_name):
        shutil.rmtree(algo.run_name)
        os.mkdir(algo.run_name)


if __name__ == '__main__':
    # run_name = input('Run name: ')
    # load_info = input('Load checkpoint info (bool name iter): ')
    # if load_info == '':
    #     load_info = ['False', '_', '-1']
    # else:
    #     load_info = load_info.split()

    run_name = 'r16_prob1_mut01'
    load_info = ['False', '_', '-1']
    algo = EGA('benchmark', 1, iterations=10000, pop_size=200,
               run_name=run_name, ckpt_period=1000, load_info=load_info, plateau_len=100,
               td_mut_scale_V=0.1)
    overwrite_save(algo)
    algo.initialize_pop()
    algo.run()

    run_name = 'r16_prob1_mut1'
    load_info = ['False', '_', '-1']
    algo = EGA('benchmark', 1, iterations=10000, pop_size=200,
               run_name=run_name, ckpt_period=1000, load_info=load_info, plateau_len=100,
               td_mut_scale_V=1)
    overwrite_save(algo)
    algo.initialize_pop()
    algo.run()

    run_name = 'r16_prob1_mut50'
    load_info = ['False', '_', '-1']
    algo = EGA('benchmark', 1, iterations=10000, pop_size=200,
               run_name=run_name, ckpt_period=1000, load_info=load_info, plateau_len=100,
               td_mut_scale_V=50)
    overwrite_save(algo)
    algo.initialize_pop()
    algo.run()

    run_name = 'r16_prob1_mut100'
    load_info = ['False', '_', '-1']
    algo = EGA('benchmark', 1, iterations=10000, pop_size=200,
               run_name=run_name, ckpt_period=1000, load_info=load_info, plateau_len=100,
               td_mut_scale_V=100)
    overwrite_save(algo)
    algo.initialize_pop()
    algo.run()
