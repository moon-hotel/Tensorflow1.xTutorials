import tensorflow as tf
import numpy as np

x = np.arange(0, 48, 1).reshape(1, 3, 4, 4)
print(x)

x = x.transpose(0, 2, 3, 1)
x = tf.convert_to_tensor(x, dtype=tf.float32)
w = tf.constant(value=1, shape=[2, 2, 3, 1], dtype=tf.float32)
b = tf.constant(value=2, shape=[1],dtype=tf.float32)
conv = tf.nn.conv2d(x, w, padding='VALID', strides=[1, 1, 1, 1])
conv = conv+b

with tf.Session() as sess:
    r = sess.run(conv)
    print(r)
    print(r.transpose(0,3,1,2))
