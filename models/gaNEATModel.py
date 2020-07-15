
import sys
sys.path.insert(0, '../problems')

from scoreProblem import ScoreProb
from actionProblem import ActionProb
from simulationProblem import SimulationProb

import numpy as np
import neat


class GANEAT:
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

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  '../saves/neatConfigs/gaNEATConfig.txt')

    def evaluate(self, genomes, config):
        for i, genome in genomes:
            genome.fitness = 0.0
            model = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.problem.test(model)
            if not isinstance(self.problem, SimulationProb):
                genome.fitness = 100 - genome.fitness

    def run(self):
        pop = neat.Population(self.config)
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(neat.StatisticsReporter())
        pop.add_reporter(neat.Checkpointer(5, filename_prefix='../saves/neatCkpts/gaNEATSave-'))

        winner = pop.run(self.evaluate, 10)
        model = neat.nn.FeedForwardNetwork.create(winner, self.config)
        steps, score = self.problem.test(model, 10)
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6))
        print('==================================================')

    def test(self, file):
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             '../saves/neatConfigs/gaNEATConfig.txt')
        pop = neat.Checkpointer.restore_checkpoint('../saves/neatCkpts/' + file)
        winner = pop.run(self.evaluate, 1)
        model = neat.nn.FeedForwardNetwork.create(winner, config)
        steps, score = self.problem.test(model, 10)
        print('==================================================')
        print('Results:   ' + str('{:.2f}'.format(steps)).zfill(7) + '   ' +
              str('{:.2f}'.format(score)).zfill(6))
        print('==================================================')


if __name__ == '__main__':
    GA = GANEAT('Score')
    GA.test('gaNEATSave-8')
