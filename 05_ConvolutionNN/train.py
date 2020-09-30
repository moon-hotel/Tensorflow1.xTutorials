from tensorflow.keras.datasets.cifar10 import load_data
from model import LeNet5
import numpy as np


def load_cifar10(shuffle=True):
    (x_train, y_train), (x_test, y_test) = load_data()
    if shuffle:
        shuffle_index = np.random.permutation(len(x_train))
        x_train, y_train = x_train[shuffle_index],y_train[shuffle_index]
    return x_train / 255., y_train, x_test / 255., y_test


def gen_batch(x, y, batch_size=64):
    s_index, e_index, batches = 0, 0 + batch_size, len(y) // batch_size
    if batches * batch_size < len(y):
        batches += 1
    for i in range(batches):
        if e_index > len(y):
            e_index = len(y)
        batch_x = x[s_index:e_index]
        batch_y = y[s_index: e_index]
        s_index, e_index = e_index, e_index + batch_size
        yield batch_x, batch_y


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_cifar10()
    model = LeNet5()
    model.fit(x_train, y_train, x_test, y_test, gen_batch)
