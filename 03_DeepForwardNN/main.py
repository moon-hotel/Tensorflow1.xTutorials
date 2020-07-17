import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.fashion_mnist import load_data


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


def forward(x, num_class):
    fc = tf.layers.dense(x, units=1024, activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    fc = tf.layers.dense(fc, units=1024, activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    fc = tf.layers.dense(fc, units=256, activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    fc = tf.layers.dense(fc, units=num_class,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    return fc


def train(x_train, y_train, x_test, y_test):
    learning_rate = 0.005
    epochs = 10
    batch_size = 256
    n = 784
    num_class = 10
    x = tf.placeholder(dtype=tf.float32, shape=[None, n], name='input_x')
    y = tf.placeholder(dtype=tf.int64, shape=[None], name="input_y")
    logits = forward(x, num_class)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy) + tf.losses.get_regularization_loss()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for step, (batch_x, batch_y) in enumerate(gen_batch(x_train, y_train, batch_size)):
                batch_x = batch_x.reshape(len(batch_y), -1)
                feed_dict = {x: batch_x, y: batch_y}
                l, acc, _ = sess.run([loss, accuracy, train_op], feed_dict=feed_dict)
                if step % 50 == 0:
                    print("Epochs[{}/{}]---Batches[{}/{}]---loss on train:{:.4}---acc: {:.4}".
                          format(epoch, epochs, step, len(x_train) // batch_size, l, acc))
            if epoch % 2 == 0:
                total_correct = 0
                for batch_x, batch_y in gen_batch(x_test, y_test, batch_size=batch_size):
                    batch_x = batch_x.reshape(len(batch_y), -1)
                    feed_dict = {x: batch_x, y: batch_y}
                    c = sess.run(correct_prediction, feed_dict=feed_dict)
                    total_correct += np.sum(c * 1.)
                print("Epochs[{}/{}]---acc on test: {:.4}".
                      format(epoch, epochs, total_correct / len(y_test)))


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_mnist_data()
    train(x_train, y_train, x_test, y_test)
