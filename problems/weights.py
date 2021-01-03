
import numpy as np
import scipy.io


class Weights:
    def __init__(self):
        # mat = scipy.io.loadmat('problems/weights_minimized.mat')
        mat = scipy.io.loadmat('problems/weights.mat')
        self.weights = mat['A']  # shape 28, 28, 16, 32

    def test(self, weights):
        diff = weights - self.weights
        return np.linalg.norm(np.expand_dims(diff.flatten(), -1), 'fro')
