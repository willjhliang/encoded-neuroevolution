import numpy as np
import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, file):
        self.file = file

    def graph_model(self):
        model = np.load('saves/' + self.file)
        W1 = model['W1']
        b1 = model['b1']
        W2 = model['W2']
        b2 = model['b2']

        # fig, axs = plt.subplots(4)
        # axs[0].plot(W1.flatten())
        # axs[1].plot(b1.flatten())
        # axs[2].plot(W2.flatten())
        # axs[3].plot(b2.flatten())
        # axs[0].imshow(W1)
        # axs[1].imshow(b1)
        # axs[2].imshow(W2)
        # axs[3].imshow(b2)
        plt.imshow(W2)
        plt.show()


if __name__ == '__main__':
    plot = Plotter('tfSave.npz')
    plot.graph_model()
