
import sys
sys.path.insert(0, 'encoders')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'problems/snake')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from egaModel import EGA

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
            ega = EGA('weights', 0, 16, 0, 0, iterations=3000,
                      run_name=run_name, ckpt_period=100,
                      load_ckpt=load_ckpt, load_name=load_name,
                      load_iter=load_iter, td_mut_scale_b=scale)
            r += ega.run()
        res.append(r / runs_per_pos)
    return res


def print_example():
    pop = np.load('saves/ega-weights-' + load_name + '/iter-' +
                  str(load_iter) + '.npy')
    ega = EGA('weights', 0, 16, 0, 0, iterations=-1,
              run_name=run_name, ckpt_period=-1,
              load_ckpt=load_ckpt, load_name=load_name,
              load_iter=load_iter)
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
    ega = EGA('weights', 0, 16, 0, 0, iterations=1000,
              run_name=run_name, ckpt_period=100,
              load_ckpt=load_ckpt, load_name=load_name,
              load_iter=load_iter)
    ega.run()


def testError():
    ega = EGA('weights', 0, 16, 0, 0, iterations=1, pop_size=2,
              run_name=run_name, ckpt_period=1,
              load_ckpt=load_ckpt, load_name=load_name,
              load_iter=load_iter)
    ind = ega.compress_decoder(ega.get_decoder(custom_data=True))
    error = ega.problem.test(ega.decode(ind)['W0'])
    return error


def stepRank(rank):
    past_pop = np.array([])
    load_iter = 0
    iterations_per = 1000
    for r in range(1, rank + 1):
        ega = EGA('weights', 0, r, 0, 0, iterations=iterations_per, pop_size=200,
                  run_name=run_name, ckpt_period=100,
                  load_ckpt=False, load_name='',
                  load_iter=load_iter, plateau_len=200)
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


if __name__ == '__main__':
    run_name = input('Run name: ')
    load_ckpt = input('Load checkpoint? (y/n) ') == 'y'
    load_name = 'null'
    load_iter = -1
    if load_ckpt:
        load_name = input('Load from: ')
        load_iter = int(input('Iteration: '))

    # ega = EGA('weights', 0, 16, 0, 0, iterations=100, pop_size=200,
    #           run_name=run_name, ckpt_period=1,
    #           load_ckpt=load_ckpt, load_name=load_name,
    #           load_iter=load_iter, mut_prob=0.3, cross_prob=0.7, par_ratio=0.3,
    #           plateau_iter=1)
    # ega.initialize_pop()
    # ega.run()
    # print(testError())
    stepRank(16)
