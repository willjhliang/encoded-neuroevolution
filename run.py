
import sys
sys.path.insert(0, 'encoders')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'problems/snake')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from egaModel import EGA

folder_name = input('Run name: ')
ega = EGA('Weights', 0, 8, 0, 0, iterations=1000,
          folder_name=folder_name, ckpt_period=1)
ega.run()
