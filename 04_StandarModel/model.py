import tensorflow as tf
import numpy as np
import os


class DeepNN():
    def __init__(self,
                 learning_rate=0.001,
                 epochs=10,
                 batch_size=256,
                 input_nodes=784,
                 hidden_1_nodes=1024,
                 hidden_2_nodes=1024,
                 hidden_3_nodes=256,
                 output_nodes=10,
                 regularization_rate=0.0001,
                 model_path='MODEL'
                 ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_nodes = input_nodes
        self.hidden_1_nodes = hidden_1_nodes
        self.hidden_2_nodes = hidden_2_nodes
        self.hidden_3_nodes = hidden_3_nodes
        self.output_nodes = output_nodes
        self.regularization_rate = regularization_rate
        self.model_path = model_path
        self._build_placeholder()
        self._build_net()

    def _build_placeholder(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_nodes], name='input_x')
        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name="input_y")

    def _build_net(self):
        with tf.name_scope('forward'):
            fc = tf.layers.dense(self.x, units=self.hidden_1_nodes, activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(self.regularization_rate))
            fc = tf.layers.dense(fc, units=self.hidden_2_nodes, activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(self.regularization_rate))
            fc = tf.layers.dense(fc, units=self.hidden_3_nodes, activation=tf.nn.relu,
                                 kernel_regularizer=tf.keras.regularizers.l2(self.regularization_rate))
            self.logits = tf.layers.dense(fc, units=self.output_nodes,
                                          kernel_regularizer=tf.keras.regularizers.l2(self.regularization_rate))
        with tf.name_scope('net_loss'):
            cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y,
                                                                           logits=self.logits)
            self.loss = tf.reduce_mean(cross_entropy) + tf.losses.get_regularization_loss()

        with tf.name_scope('accuracy'):
            self.correct_prediction = tf.equal(tf.argmax(self.logits, axis=1), self.y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

    def train(self, x_train, y_train, x_test, y_test, gen_batch):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
        model_name = "model"
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            start_epoch = 0
            check_point = tf.train.latest_checkpoint(self.model_path)
            if check_point:
                saver.restore(sess, check_point)
                # acc = self.evaluate(sess, x_test, y_test, gen_batch)
                # print("==================acc",acc)
                # 这里可以在载入模型后，立即进行准确率计算，可以比较与保存的时候是否一样
                start_epoch += int(check_point.split('-')[-1])
                print("### Loading exist model <{}> successfully...".format(check_point))
            for epoch in range(start_epoch, self.epochs):
                for step, (batch_x, batch_y) in enumerate(gen_batch(x_train, y_train, self.batch_size)):
                    batch_x = batch_x.reshape(len(batch_y), -1)
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
            batch_x = batch_x.reshape(len(batch_y), -1)
            feed_dict = {self.x: batch_x, self.y: batch_y}
            c = sess.run(self.correct_prediction, feed_dict=feed_dict)
            total_correct += np.sum(c * 1.)
        acc = total_correct / len(y_test)
        return acc
