
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'snake')

from egaModel import EGA

# print('AR(2) p=0.1')
# EGA('Simulation', 2, 0, 0, 0, p=0.1, c=0.9).run()
print('TD(2) p=0.01')
EGA('Action', 0, 2, 0, 0, p=0.01, c=0.9, iterations=50, file_name='pixel').run()
# print('TD(2) p=0.1')
# EGA('Simulation', 0, 2, 0, 0, p=0.1, c=0.9).run()
# print('TD(2) p=1')
# EGA('Simulation', 0, 2, 0, 0, p=1, c=0.9).run()
# print('TD(2) p=10')
# EGA('Simulation', 0, 2, 0, 0, p=10, c=0.9).run()
