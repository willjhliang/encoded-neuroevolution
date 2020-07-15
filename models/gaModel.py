from ..problems.scoreProblem import ScoreProb
from ..problems.actionProblem import ActionProb
from ..problems.simulationProblem import SimulationProb

import numpy as np


class GA:
    def __init__(self, problem, iterations=100, pop_size=100, mut_prob=0.2,
                 elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3, **kwargs):
        self.iterations = iterations
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)

        if problem == 'Score':
            self.problem = ScoreProb()
        if problem == 'Action':
            self.problem = ActionProb()
        if problem == 'Simulation':
            self.problem = SimulationProb(kwargs['p'], kwargs['c'])
        self.ln1, self.ln2, self.ln3 = self.problem.model_dims()
        self.compress_len = self.compress(self.problem.model()).size

        self.clip_lo = -1
        self.clip_hi = 0

    def compress(self, model):
        # W1 = model['W1']
        # b1 = model['b1']

        # ret = W1.flatten()
        # ret = np.concatenate((ret, b1.flatten()))
        # return ret

        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']
        ret = W1.flatten()
        ret = np.concatenate((ret, b1.flatten()))
        ret = np.concatenate((ret, W2.flatten()))
        ret = np.concatenate((ret, b2.flatten()))

        return ret

    def expand(self, arr):
        # W1, b1 = np.split(arr, [1 * 4])
        # W1 = W1.reshape(1, 4)
        # b1 = b1.reshape(1, 1)
        # return {'W1': W1, 'b1': b1}

        W1, arr = np.split(arr, [self.ln2 * self.ln1])
        b1, arr = np.split(arr, [self.ln2])
        W2, b2 = np.split(arr, [self.ln3 * self.ln2])
        W1 = W1.reshape(self.ln2, self.ln1)
        b1 = b1.reshape(self.ln2, 1)
        W2 = W2.reshape(self.ln3, self.ln2)
        b2 = b2.reshape(self.ln3, 1)

        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def clip(self, x):
        # x[x < self.clip_lo] = self.clip_lo
        # x[x > self.clip_hi] = self.clip_hi

        return x

    def cross(self, x, y):
        c1 = x.copy()
        c2 = y.copy()
        for i in range(self.compress_len):
            if np.random.random() < 0.5:
                c1[i] = y[i].copy()
                c2[i] = x[i].copy()
        c1 = self.clip(c1)
        c2 = self.clip(c2)
        return c1, c2

    def mut(self, c):
        for i in range(self.compress_len):
            if np.random.random() < self.mut_prob:
                c[i] += np.random.normal(scale=0.1)
        c = self.clip(c)
        return c

    def run(self):
        pop = np.array([np.zeros(self.compress_len)] * self.pop_size)
        for p in range(self.pop_size):
            pop[p] = self.compress(self.problem.model())
            pop[p] = self.clip(pop[p])

        for t in range(1, self.iterations + 1):
            fitness = np.zeros(self.pop_size)
            for p in range(self.pop_size):
                J = self.problem.test(self.expand(pop[p]))[0]
                fitness[p] = J
            pop = pop[np.argsort(fitness)]
            fitness = fitness[np.argsort(fitness)]

            J = self.problem.test(self.expand(pop[0]))[0]
            if t % 10 == 0:
                print('Iter ' + str(t).zfill(3) + ': ' + str(J))

            prob = np.array(fitness, copy=True)
            prob = prob / np.sum(prob)
            cum_prob = np.cumsum(prob)
            par = np.array([np.zeros(self.compress_len)] * self.par_size)
            for i in range(self.elite_size):
                par[i] = pop[i].copy()
            for i in range(self.elite_size, self.par_size):
                idx = np.searchsorted(cum_prob, np.random.random())
                par[i] = pop[idx].copy()

            eff = np.array([False] * self.par_size)
            par_ct = 0
            while par_ct < 1:
                for i in range(self.par_size):
                    if np.random.random() < self.cross_prob:
                        eff[i] = True
                        par_ct += 1
            eff_par = par[eff].copy()

            pop = np.array([np.zeros(self.compress_len)] * self.pop_size)
            for i in range(self.par_size):
                pop[i] = par[i].copy()
            for i in range(self.par_size, self.pop_size, 2):
                p1 = eff_par[np.random.randint(0, par_ct)].copy()
                p2 = eff_par[np.random.randint(0, par_ct)].copy()
                pop[i], pop[i + 1] = self.cross(p1, p2)
                pop[i] = self.mut(pop[i])
                pop[i + 1] = self.mut(pop[i + 1])
        model = self.expand(pop[0])
        steps, score = self.problem.test(model, 1000)
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6))
        print('==================================================')
        np.savez('../saves/gaSave.npz', W1=model['W1'], b1=model['b1'], W2=model['W2'], b2=model['b2'])


if __name__ == '__main__':
    GA = GA()
    GA.run()
