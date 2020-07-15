
import sys
sys.path.insert(0, '../problems')

from scoreProblem import ScoreProb
from actionProblem import ActionProb
from simulationProblem import SimulationProb

import numpy as np
import neat


class EGANEATCPPN:
    def __init__(self, problem, solver_type, iterations=10, **kwargs):
        self.iterations = iterations

        if problem == 'Score':
            self.problem = ScoreProb()
        if problem == 'Action':
            self.problem = ActionProb()
        if problem == 'Simulation':
            self.problem = SimulationProb(kwargs['p'], kwargs['c'])
        self.ln1, self.ln2, self.ln3 = self.problem.model_dims()

        self.solver_type = solver_type
        if self.solver_type == 'NEAT':
            k = 'egaNEATConfig.txt'
        if self.solver_type == 'CPPN':
            k = 'egaCPPNConfig.txt'
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  '../saves/neatConfigs/' + k)

    def model(self, net):
        L1 = np.zeros((self.ln2, self.ln1 + 1))
        L2 = np.zeros((self.ln3, self.ln2 + 1))
        for i in range(L1.shape[0]):
            for j in range(L1.shape[1]):
                X = np.array([i, j, 0])
                L1[i][j] = net.activate(np.expand_dims(X.T, axis=-1))[0]
        for i in range(L2.shape[0]):
            for j in range(L2.shape[1]):
                X = np.array([i, j, 1])
                L2[i][j] = net.activate(np.expand_dims(X.T, axis=-1))[0]
        W1 = L1[:, :-1]
        b1 = np.expand_dims(L1[:, -1], axis=-1)
        W2 = L2[:, :-1]
        b2 = np.expand_dims(L2[:, -1], axis=-1)
        return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    def evaluate(self, genomes, config):
        for i, genome in genomes:
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            model = self.model(net)
            genome.fitness = self.problem.test(model)
            if not isinstance(self.problem, SimulationProb):
                genome.fitness = 100 - genome.fitness

    def run(self):
        pop = neat.Population(self.config)
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(neat.StatisticsReporter())
        pop.add_reporter(neat.Checkpointer(5, filename_prefix='../saves/neatCkpts/ega' + self.solver_type + 'Save-'))

        winner = pop.run(self.evaluate, self.iterations)
        net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        steps, score = self.problem.test(self.model(net))
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6))
        print('==================================================')

    def test(self, file):
        pop = neat.Checkpointer.restore_checkpoint('../saves/neatCkpts/' + file)
        winner = pop.run(self.evaluate, 1)
        net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        steps, score = self.problem.test(self.model(net), 1000)
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6))
        print('==================================================')


if __name__ == '__main__':
    GA = EGANEATCPPN('Score', 'CPPN')
    GA.run()
