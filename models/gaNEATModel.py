from scoreModel import ScoreModel
from random import randint
import numpy as np
from collections import Counter
import neat


class GANEAT:
    def __init__(self, iterations=100, pop_size=100, mut_prob=0.2, elite_ratio=0.01, cross_prob=0.5, par_ratio=0.3):
        self.iterations = iterations
        self.pop_size = pop_size
        self.mut_prob = mut_prob
        self.elite_ratio = elite_ratio
        self.cross_prob = cross_prob
        self.par_ratio = par_ratio
        self.par_size = (int)(self.par_ratio * self.pop_size)
        self.elite_size = (int)(self.elite_ratio * pop_size)
        self.clip_lo = -1
        self.clip_hi = 0

    def clip(self, x):
        x[x < self.clip_lo] = self.clip_lo
        x[x > self.clip_hi] = self.clip_hi
        return x

    def evaluate(self, genomes, config):
        for i, genome in genomes:
            genome.fitness = 0.0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            for i in range(self.X.shape[1]):
                output = net.activate(self.X[:, i])
                genome.fitness += (output[0] - self.y[i]) ** 2
            genome.fitness *= 1.0 / self.X.shape[1]
            genome.fitness = 1 - genome.fitness

    def run(self):
        nn = ScoreModel()
        self.training_data = nn.initial_population()
        self.X = np.array([i[0] for i in self.training_data]).T
        self.y = np.array([i[1] for i in self.training_data]).T

        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             '../saves/neatConfigs/gaNEATConfig.txt')
        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(neat.StatisticsReporter())
        pop.add_reporter(neat.Checkpointer(5, filename_prefix='../saves/neatCkpts/neat-checkpoint-'))

        winner = pop.run(self.evaluate, 10)
        model = neat.nn.FeedForwardNetwork.create(winner, config)
        nn.test_neat_model(model)

    def test(self, file):
        nn = ScoreModel()
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             '../saves/neatConfigs/gaNEATConfig.txt')
        pop = neat.Checkpointer.restore_checkpoint('../saves/neatCkpts/' + file)
        winner = pop.statistics.best_genome()
        model = neat.nn.FeedForwardNetwork.create(winner, config)
        nn.test_neat_model(model)


if __name__ == '__main__':
    GA = GANEAT()
    GA.run()
