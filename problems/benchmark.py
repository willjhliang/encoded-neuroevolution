
import numpy as np
import scipy.io
import tensorly
from cec2013lsgo.cec2013 import Benchmark


class BM:
    def __init__(self):
        bench = Benchmark()
        problem_num = 1
        self.eval = bench.get_function(problem_num)
        self.scale = bench.get_info(problem_num)['upper']

    def test(self, vals, print_stats=False):
        fitness = self.eval(vals * self.scale)  # Scale [-1, 1] up to problem lo and hi

        if print_stats:
            print('Fitness: ' + str(fitness))

        return fitness
