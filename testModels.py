
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'snake')

from egaModel import EGA
import numpy as np

file_name = input('Run name: ')
ega = EGA('Action', 0, 8, 0, 0, iterations=1000000,
          file_name=file_name)
ega.run()
