
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

model_type = sys.argv[1]  # Command-line argument


def load_data():
    (train, test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True
    )

    def normalize(image, label):
        return tf.cast(image, tf.float32) / 255., label

    train = train.map(
        normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train = train.cache()
    train = train.shuffle(ds_info.splits['train'].num_examples)
    train = train.batch(128)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)

    test = test.map(
        normalize, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = test.cache()
    test = test.batch(128)
    test = test.prefetch(tf.data.experimental.AUTOTUNE)

    return train, test


def create_model(model_type):
    if model_type == 'FCNN':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        print(model.summary())
    if model_type == 'CNN':
        model = tf.keras.models.Sequential([
            layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(28, 28, 1), kernel_regularizer='l2'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer='l2'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer='l2'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10)
        ])
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        print(model.summary())
    return model



def train_model(model_type, model):
    train, test = load_data()
    if model_type == 'FCNN':
        epochs = 10
        history = model.fit(train, epochs=epochs, validation_data=test)
    if model_type == 'CNN':
        epochs = 20
        history = model.fit(train, epochs=epochs, validation_data=test)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    plt.show()

    return model


def load_model(model_type, model):
    model.load_weights('saves/' + model_type.lower() + '-mnist/weights')
    return model


def save_model(model_type, model):
    model.save_weights('saves/' + model_type.lower() + '-mnist/weights')

    weights = {}
    for layer in model.layers:
        w = layer.get_weights()
        if len(w) > 0:
            weights[layer.name + '_W'] = w[0]
            weights[layer.name + '_b'] = w[1]
    np.savez('saves/' + model_type.lower() + '-mnist/weights', **weights)


model = create_model(model_type)
# model = train_model(model_type, model)
# save_model(model_type, model)
