
import numpy as np
import tensorflow as tf


class MNIST:
    def __init__(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        self.x = np.expand_dims(np.concatenate((x_train, x_test)) / 255.0, axis=-1)
        self.y = np.concatenate((y_train, y_test))

    def test(self, model):
        loss = model.evaluate(self.x, self.y, verbose=0)
        return loss[0], loss[1]
