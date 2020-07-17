### 1 前言

在上一篇文章中，我们介绍了如何自己编码实现交叉熵损失的计算。但其实在Tensoflow中已经实现了这个方法，并且还实现了两种（单目标分类和多目标分类）分类情况下交叉熵损失的计算。下面的代码就是我们自己实现交叉熵的计算方法：

```python
def softmax_cross_entropy(labels, logits):
    soft_logits = tf.nn.softmax(logits)
    soft_logits = tf.clip_by_value(soft_logits, 0.000001, 0.999999)
    cross_entropy = -tf.reduce_sum(labels * tf.log(soft_logits), axis=1)
    return cross_entropy
```

当然，完整的说应该是同时实现了softmax和交叉熵操作。由于这两个操作使用得太频繁，几乎已经成为了分类任务中的标准，所有Tensorflow提供了两个函数来进行实现，分别是`softmax_cross_entropy_with_logits_v2`和 `sparse_softmax_cross_entropy_with_logit`，后文分别简称`softmax_cross`和`sparse_softmax`。

既然两个函数都能实现softmax+cross entropy的操作，那为什么还有两个呢？它们的区别在哪儿?

### 2 相同点

两者都是先经过softmax处理，然后来计算交叉熵，并且最终的结果是一样的，再强调一遍，**最终结果都一样**。那既然有了`softmax_cross` 这个方法，那么`sparse_softmax` 有何用？原因在于，在单目标的任务分类场景中（例如Fashion MNIST分类任务中，每个图片都只有一个类标标签），TensorFlow提供了`sparse_softmax` 函数来进一步的加速计算过程。

### 3 异同点

不同点在于两者在传递参数时的形式上。

对于`softmax_cross`来说，其`logits=` 的维度是`[batch_size,num_classes]`，即正向传播最后的输出层结果（unscaled log probability）。同时，`labels=`的维度也是`[batch_size,num_classes]`，即正确标签的one_hot形式。

对于`sparse_softmax`来说，其`logits=` 的维度是`[batch_size,num_classes]`，即正向传播最后的输出层结果；但`labels=`的维度有所不同，此时就不再是one_hot形式的标签，而是每一个标签所代表的真实类别，其维度为`[batch_size]`的一个一维张量。

同时，需要注意的是为了避免不必要的混淆，Tensorflow要求在使用这两个函数时，在传递参数的过程中必须加上形参的参数名，即`softmax_cross(labels=,logits=)`这样的形式。

### 4 结果比较

```python
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
    
#结果
1.0374309
1.0374309
1.0374309
```

### 引用

[1]https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits

[2]https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/nn/softmax_cross_entropy_with_logits_v2



## [<返回主页>](../README.md)