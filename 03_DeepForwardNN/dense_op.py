import tensorflow as tf
import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6.]])
b = np.array([1, 1, 1, 0, 2, 1.]).reshape(3, 2)
print(np.matmul(a, b))

x = tf.Variable(tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32))
weight = np.array([1, 1, 1, 0, 2, 1.]).reshape(3, 2)

fc = tf.layers.dense(x, units=2, activation=tf.nn.relu,
                     kernel_initializer=tf.constant_initializer(value=weight),
                     kernel_regularizer=tf.keras.regularizers.l2(0.1))
l = tf.losses.get_regularization_loss()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    f, loss = sess.run([fc, l])
    print(f)
    print(loss)
