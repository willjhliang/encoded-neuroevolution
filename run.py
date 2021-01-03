
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
                      folder_name=folder_name, ckpt_period=100,
                      load_ckpt=load_ckpt, load_name=load_name,
                      load_iter=load_iter, td_mut_scale_b=scale)
            r += ega.run()
        res.append(r / runs_per_pos)
    return res


def print_example():
    pop = np.load('saves/ega-weights-' + load_name + '/iter-' +
                  str(load_iter) + '.npy')
    ega = EGA('weights', 0, 16, 0, 0, iterations=-1,
              folder_name=folder_name, ckpt_period=-1,
              load_ckpt=load_ckpt, load_name=load_name,
              load_iter=load_iter)
    res = ega.decode(pop[0])
    ega.test(pop)
    return res['W0']

def profileCall():
    # Make sure to set problem as MINIMIZED weights
    ega = EGA('weights', 0, 16, 0, 0, iterations=1000,
              folder_name=folder_name, ckpt_period=100,
              load_ckpt=load_ckpt, load_name=load_name,
              load_iter=load_iter)
    ega.run()

folder_name = input('Run name: ')
load_ckpt = input('Load checkpoint? (y/n) ') == 'y'
load_name = 'null'
load_iter = '-1'
if load_ckpt:
    load_name = input('Load from: ')
    load_iter = int(input('Iteration: '))

if folder_name == 'print':
    print(print_example())
elif folder_name == 'profile':
    profile = cProfile.Profile()
    profile.runcall(profileCall)
    ps = pstats.Stats(profile)
    ps.print_stats()
else:
    # print(optimize_td_mut_scale())
    ega = EGA('weights', 0, 16, 0, 0, iterations=1000,
              folder_name=folder_name, ckpt_period=100,
              load_ckpt=load_ckpt, load_name=load_name,
              load_iter=load_iter, mut_prob=0.4, cross_prob=0.9)
    ega.run()
