
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'problems/snake')

from egaModel import EGA
import numpy as np

file_name = input('Run name: ')
ega = EGA('MNIST', 0, 8, 0, 0, iterations=100,
          file_name=file_name, ckpt_period=1)
ega.run()
