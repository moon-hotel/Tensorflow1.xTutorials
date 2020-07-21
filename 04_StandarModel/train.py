from tensorflow.keras.datasets.fashion_mnist import load_data
from model import DeepNN


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = load_data()
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
    x_train, y_train, x_test, y_test = load_mnist_data()
    model = DeepNN(learning_rate=0.001,
                   epochs=10,
                   batch_size=256,
                   input_nodes=784,
                   hidden_1_nodes=1024,
                   hidden_2_nodes=1024,
                   hidden_3_nodes=256,
                   output_nodes=10,
                   regularization_rate=0.0001,
                   model_path='MODEL')
    model.train(x_train, y_train, x_test, y_test, gen_batch)
