
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'snake')

from egaModel import EGA
import numpy as np

file_name = input()
# f = np.load('saves/egaPop-' + file_name + '.npz')
# print(f['iterations'])
# pop = f['pop']
# food_arr = f['food']
a = EGA('Action', 0, 8, 0, 0, iterations=300, file_name=file_name)
a.run()
# a.test(pop, food_arr)
