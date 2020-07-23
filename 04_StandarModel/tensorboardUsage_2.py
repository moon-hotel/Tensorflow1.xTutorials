from tensorflow.keras.datasets.fashion_mnist import load_data
import tensorflow as tf
import numpy as np


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


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def train(x_train, y_train, x_test, y_test):
    learning_rate = 0.005
    epochs = 10
    batch_size = 256
    n = 784
    num_class = 10
    x = tf.placeholder(dtype=tf.float32, shape=[None, n], name='input_x')
    y = tf.placeholder(dtype=tf.int64, shape=[None], name="input_y")
    with tf.name_scope('layers'):
        with tf.name_scope('fc1'):
            fc = tf.layers.dense(x, units=1024, activation=tf.nn.relu, name='fc1')
            with tf.variable_scope("fc1", reuse=True):
                w1 = tf.get_variable("kernel")
            variable_summaries(w1)
        with tf.name_scope('fc2'):
            logits = tf.layers.dense(fc, units=num_class, name='fc2')
            with tf.variable_scope("fc2", reuse=True):
                w2 = tf.get_variable("kernel")
            variable_summaries(w2)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)
    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, axis=1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('logs/', sess.graph)

        for epoch in range(epochs):
            for step, (batch_x, batch_y) in enumerate(gen_batch(x_train, y_train, batch_size)):
                batch_x = batch_x.reshape(len(batch_y), -1)
                feed_dict = {x: batch_x, y: batch_y}
                l, acc, _, summary = sess.run([loss, accuracy, train_op, merged], feed_dict=feed_dict)
                if step % 50 == 0:
                    print("Epochs[{}/{}]---Batches[{}/{}]---loss on train:{:.4}---acc: {:.4}".
                          format(epoch, epochs, step, len(x_train) // batch_size, l, acc))
                writer.add_summary(summary, epoch * (len(x_train) // batch_size) + step)
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
