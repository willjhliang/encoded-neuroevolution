
import sys
sys.path.insert(0, 'encoders')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'problems/snake')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from egaModel import EGA


def optimize_td_mut_scale(runs_per_pos=5):
    pos = [10 ** a for a in range(0, -8, -1)]
    print(pos)
    res = []
    for scale in pos:
        print('------------------------------')
        print('SCALE: ' + str(scale))
        r = 0
        for i in range(runs_per_pos):
            ega = EGA('weights', 0, 8, 0, 0, iterations=100,
                      folder_name=folder_name, ckpt_period=10,
                      load_ckpt=load_ckpt, load_name=load_name,
                      load_iter=load_iter, td_mut_scale=scale)
            r += ega.run()
        res.append(r / runs_per_pos)
    return res


def print_example():
    pop = np.load('saves/ega-weights-' + load_name + '/iter-' +
                  str(load_iter) + '.npy')
    ega = EGA('weights', 0, 16, 0, 0, iterations=-1,
              folder_name=folder_name, ckpt_period=-1,
              load_ckpt=load_ckpt, load_name=load_name,
              load_iter=load_iter, td_mut_scale=0.000001)
    res = ega.decode(pop[0])
    ega.test(pop)
    return res['W0']


folder_name = input('Run name: ')
load_ckpt = input('Load checkpoint? (y/n) ') == 'y'
load_name = 'null'
load_iter = '-1'
if load_ckpt:
    load_name = input('Save name: ')
    load_iter = int(input('Iteration: '))

if folder_name == 'print':
    print(print_example())
else:
    ega = EGA('weights', 0, 24, 0, 0, iterations=3000,
              folder_name=folder_name, ckpt_period=100,
              load_ckpt=load_ckpt, load_name=load_name,
              load_iter=load_iter, td_mut_scale=0.00000001)
    ega.run()
