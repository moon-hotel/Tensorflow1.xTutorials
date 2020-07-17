import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.fashion_mnist import load_data


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def load_mnist_data(one_hot=True):
    (x_train, y_train), (x_test, y_test) = load_data()
    if one_hot:
        y_train = dense_to_one_hot(y_train)
        y_test = dense_to_one_hot(y_test)
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


def forward(x, w, b):
    f = tf.matmul(x, w) + b
    return f


def softmax_cross_entropy(labels, logits):
    soft_logits = tf.nn.softmax(logits)
    soft_logits = tf.clip_by_value(soft_logits, 0.000001, 0.999999)
    cross_entropy = -tf.reduce_sum(labels * tf.log(soft_logits), axis=1)
    return cross_entropy


def train(x_train, y_train, x_test, y_test):
    learning_rate = 0.005
    epochs = 10
    batch_size = 256
    n = 784
    num_class = 10
    x = tf.placeholder(dtype=tf.float32, shape=[None, n], name='input_x')
    y = tf.placeholder(dtype=tf.float32, shape=[None, num_class], name="input_y")
    w = tf.Variable(tf.truncated_normal(shape=[n, num_class],
                                        mean=0, stddev=0.1,
                                        dtype=tf.float32))
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[num_class]))
    y_pred = forward(x, w, b)
    cross_entropy = softmax_cross_entropy(y, y_pred)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(y, y_pred)
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            for step, (batch_x, batch_y) in enumerate(gen_batch(x_train, y_train, batch_size=batch_size)):
                batch_x = batch_x.reshape(len(batch_y), -1)
                feed_dict = {x: batch_x, y: batch_y}
                l, acc, _ = sess.run([loss, accuracy, train_op], feed_dict=feed_dict)
                if step % 100 == 0:
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
