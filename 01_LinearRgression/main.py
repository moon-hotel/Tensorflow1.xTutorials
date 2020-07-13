import tensorflow as tf
from tensorflow.keras.datasets.boston_housing import load_data
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_boston_data():
    (x_train, y_train), (x_test, y_test) = load_data(test_split=0.3)
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    return x_train, y_train, x_test, y_test

def forward(x, w, b):
    return tf.matmul(x, w) + b


def MSE(y_true, y_pred):
    return 0.5*tf.reduce_mean(tf.square(y_true - y_pred))


def train(x_train, y_train, x_test, y_test):
    learning_rate = 0.1
    epochs = 300
    m, n = x_train.shape
    x = tf.placeholder(dtype=tf.float32, shape=[None,n],name='input_x')
    y = tf.placeholder(dtype=tf.float32, shape=[None],name="input_y")
    w = tf.Variable(tf.truncated_normal(shape=[n, 1],
                                        mean=0, stddev=0.1,
                                        dtype=tf.float32))
    b = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[1]))
    y_pred = forward(x, w, b)
    loss = MSE(y, y_pred)

    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            feed_dict = {x: x_train, y: y_train}
            l, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            print("[{}/{}]----loss on train:{:.4}".format(epoch, epochs, l))
            if epoch % 10 == 0:
                feed_dict = {x: x_test, y: y_test}
                l = sess.run(loss, feed_dict=feed_dict)
                print("[{}/{}]----loss on test:{:.4}-----RMSE: {:.4}".
                      format(epoch, epochs, l, np.sqrt(l)))

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_boston_data()
    train(x_train, y_train, x_test, y_test)
