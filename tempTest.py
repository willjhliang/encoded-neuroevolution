
import sys
sys.path.insert(0, 'models')
sys.path.insert(0, 'problems')
sys.path.insert(0, 'snake')

# from bpModel import BP

# bp = BP('Score')
# bp.test()

# from actionProblem import ActionProb

# a = ActionProb()
# a.get_training_data('')

from egaModel import EGA

a = EGA('Action', 0, 2, 0, 0, iterations=100, file_name='default').tf_model()
