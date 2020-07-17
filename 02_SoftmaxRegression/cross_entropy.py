import tensorflow as tf

logits = tf.constant([[0.5, 0.3, 0.6], [0.5, 0.4, 0.3]], dtype=tf.float32)
y_true = tf.constant([[0, 0, 1], [0, 1, 0]], dtype=tf.float32)

logits_soft = tf.nn.softmax(logits)
cross_entropy_1 = tf.reduce_sum(y_true * tf.log(logits_soft), axis=1)
loss_1 = -tf.reduce_mean(cross_entropy_1)

cross_entropy_2 = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=logits)
loss_2 = tf.reduce_mean(cross_entropy_2)

cross_entropy_3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_true, 1), logits=logits)
loss_3 = tf.reduce_mean(cross_entropy_3)

with tf.Session() as sess:
    l1, l2, l3 = sess.run([loss_1, loss_2, loss_3])
    print(l1)
    print(l2)
    print(l3)
