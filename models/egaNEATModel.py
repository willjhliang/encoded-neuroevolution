from scoreModel import ScoreModel
import numpy as np
import neat


class EGANEAT:
    def __init__(self, iterations=10):
        self.iterations = iterations

        self.nn = ScoreModel()
        self.ln1, self.ln2, self.ln3 = self.nn.model_dims()
        self.training_data = self.nn.initial_population()
        self.X = np.array([i[0] for i in self.training_data]).T
        self.y = np.array([i[1] for i in self.training_data]).T

        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             '../saves/neatConfigs/egaNEATConfig.txt')

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
            for i in range(self.X.shape[1]):
                genome.fitness += self.nn.forward_prop(self.X[:, i], self.y[i], model)[0]
            genome.fitness *= 1.0 / self.X.shape[1]
            genome.fitness = 1 - genome.fitness

    def run(self):
        pop = neat.Population(self.config)
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(neat.StatisticsReporter())
        pop.add_reporter(neat.Checkpointer(5, filename_prefix='../saves/neatCkpts/egaNEATSave-'))

        winner = pop.run(self.evaluate, self.iterations)
        net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        self.nn.test_model(self.model(net))

    def test(self, file):
        pop = neat.Checkpointer.restore_checkpoint('../saves/neatCkpts/' + file)
        winner = pop.run(self.evaluate, 1)
        net = neat.nn.FeedForwardNetwork.create(winner, self.config)
        self.nn.test_model(self.model(net))


if __name__ == '__main__':
    GA = EGANEAT()
    GA.run()
