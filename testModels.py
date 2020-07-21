
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'snake')

from egaModel import EGA

EGA('Action', 0, 2, 0, 0, p=0.01, c=0.9, iterations=100, file_name='400-input').run()
