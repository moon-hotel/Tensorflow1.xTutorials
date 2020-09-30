import tensorflow as tf
import os
import numpy as np


class LeNet5():
    def __init__(self,
                 image_size=32,
                 channles=3,
                 filter_size=5,
                 learning_rate=0.002,
                 epochs=50,
                 batch_size=128,
                 num_class=10,
                 model_path='MODEL',
                 ):
        self.image_size = image_size
        self.channles = channles
        self.filter_size = filter_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_class = num_class
        self.model_path = model_path
        self._build_placeholder()
        self._build_lenet5()

    def _build_placeholder(self):
        self.x = tf.placeholder(dtype=tf.float32,
                                shape=[None, self.image_size, self.image_size, self.channles],
                                name='input_x')  # [n,32,32,3]
        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name='input_y')

    def _build_lenet5(self):
        with tf.name_scope('layer1'):
            filters = tf.Variable(tf.truncated_normal(shape=[self.filter_size,
                                                             self.filter_size,
                                                             self.channles,
                                                             6], mean=0, stddev=0.1))
            bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[6]))
            conv1 = tf.nn.conv2d(self.x, filters, strides=1, padding='VALID')  # [n,28,28,16]
            conv1 = tf.nn.bias_add(conv1, bias)
            conv1 = tf.nn.relu(conv1)
            # print(conv1)
            pool1 = tf.nn.max_pool2d(conv1, ksize=2, strides=2, padding='VALID')  # [n,14,14,16]
            # print(pool1)
        with tf.name_scope('layer2'):
            filters = tf.Variable(tf.truncated_normal(shape=[self.filter_size,
                                                             self.filter_size,
                                                             6,
                                                             16], mean=0, stddev=0.1))
            bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[16]))
            conv2 = tf.nn.conv2d(pool1, filters, strides=1, padding='VALID')  # [n,10,10,32]
            conv2 = tf.nn.bias_add(conv2, bias)
            # print(conv2)
            conv2 = tf.nn.relu(conv2)
            pool2 = tf.nn.max_pool2d(conv2, ksize=2, strides=2, padding='VALID')  # [n,5,5,32]
            # print(pool2)
        pool_shape = pool2.shape.as_list()
        fc_input = tf.reshape(pool2, [-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])
        with tf.name_scope('layer3'):
            fc = tf.layers.dense(fc_input, units=120, activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0001))
            fc = tf.layers.dense(fc, units=84, activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0001))
            self.logits = tf.layers.dense(fc, units=self.num_class, kernel_regularizer=tf.keras.regularizers.l2(0.0001))

        with tf.name_scope('net_loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy) + tf.losses.get_regularization_loss()
        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), self.y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def fit(self, x_train, y_train, x_test, y_test, gen_batch):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
        model_name = "model"
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            tf.summary.FileWriter('logs/', sess.graph)
            start_epoch = 0
            check_point = tf.train.latest_checkpoint(self.model_path)
            if check_point:
                saver.restore(sess, check_point)
                start_epoch += int(check_point.split('-')[-1])
                print("### Loading exist model <{}> successfully...".format(check_point))
            for epoch in range(start_epoch, self.epochs):
                for step, (batch_x, batch_y) in enumerate(gen_batch(x_train, y_train, self.batch_size)):
                    batch_y = batch_y.reshape(-1)
                    feed_dict = {self.x: batch_x, self.y: batch_y}
                    l, acc, _ = sess.run([self.loss, self.accuracy, train_op], feed_dict=feed_dict)
                    if step % 50 == 0:
                        print("Epochs[{}/{}]---Batches[{}/{}]---loss on train:{:.4}---acc: {:.4}".
                              format(epoch, self.epochs, step, len(x_train) // self.batch_size, l, acc))
                if epoch % 2 == 0:
                    acc = self.evaluate(sess, x_test, y_test, gen_batch)
                    saver.save(sess, os.path.join(self.model_path, model_name + "_{}".format(acc)),
                               global_step=epoch, write_meta_graph=False)
                    print("### Saving  model successfully...")
                    print("Epochs[{}/{}]---acc on test: {:.4}".
                          format(epoch, self.epochs, acc))

    def evaluate(self, sess, x_test, y_test, gen_batch):
        total_correct = 0
        for batch_x, batch_y in gen_batch(x_test, y_test, self.batch_size):
            batch_y = batch_y.reshape(-1)
            feed_dict = {self.x: batch_x, self.y: batch_y}
            c = sess.run(self.correct_prediction, feed_dict=feed_dict)
            total_correct += np.sum(c * 1.)
        acc = total_correct / len(y_test)
        return acc


if __name__ == '__main__':
    model = LeNet5()
