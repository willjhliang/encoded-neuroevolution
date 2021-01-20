
import numpy as np
import scipy.io
import tensorly


class Weights:
    def __init__(self):
        # mat = scipy.io.loadmat('problems/weights_minimized.mat')
        mat = scipy.io.loadmat('problems/weights.mat')
        self.weights = mat['A']  # shape 28, 28, 16, 32
        self.norm = np.linalg.norm(tensorly.base.unfold(self.weights, 0),
                                   'fro')

    def test(self, weights):
        diff = weights - self.weights

        res = np.linalg.norm(tensorly.base.unfold(diff, 0), 'fro')
        return res / self.norm
