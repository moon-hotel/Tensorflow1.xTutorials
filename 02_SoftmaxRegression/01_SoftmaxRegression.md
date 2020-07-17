## 1 前言

在上一篇文章中笔者介绍了如何通过Tensorflow来实现线性回归。在接下来的这篇文章中，笔者将会以Fashion MNIST数据集为例来介绍如何用Tensorflow实现一个Softmax多分类模型。在这篇文章中，我们会开始慢慢接触到Tensoflow中用于实现分类模型的API，例如`tf.nn.softmax()`，`softmax_cross_entropy_with_logits_v2`等。

## 2 数据处理

### 2.1 导入相关包

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.fashion_mnist import load_data
```

正如上一篇文章中说到，`tensorflow.keras.datasets`中内置了一些比较常用的数据集，并且Tensorflow在各个数据集中均实现了`load_data`这个方法来对数据集进行载入。所以，上面第三行代码的作用就是用来载入fashion_mnist数据集。

### 2.2 载入数据

- 标签转换

  在载入数据前先介绍一个将按序类别编码转化为one-hot编码的方法：

  ```python
  def dense_to_one_hot(labels_dense, num_classes=10):
      num_labels = labels_dense.shape[0]
      index_offset = np.arange(num_labels) * num_classes
      labels_one_hot = np.zeros((num_labels, num_classes))
      labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
      return labels_one_hot
  ```

  该函数的作用就是将普通的`[0,1,2,3]`类别标签转换为one-hot的编码形式。

- 载入数据集

  载入任务需要的fashion mnist数据集：

  ```python
  def load_mnist_data(one_hot=True):
      (x_train, y_train), (x_test, y_test) = load_data()
      if one_hot:
          y_train = dense_to_one_hot(y_train)
          y_test = dense_to_one_hot(y_test)
      return x_train / 255., y_train, x_test / 255., y_test
  ```

- 批产生器

  构造一个batch迭代器，在训练的过程中每个返回一个batch的数据：

  ```python
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
  ```

  其中第6、7行的代码表示，当计算得到最后一个batch的结束索引大于数据集长度时，则只取到最后一个样本即可，也就是说，最后一个batch的样本数可能没有batch size个。例如100个样本，batch size 为40，那么每个batch的样本数分别为：40，40和20。

## 3 框架介绍

### 3.1 定义正向传播

```python
def forward(x, w, b):
    f = tf.matmul(x, w) + b
    return f
```

### 3.2 定义损失

```python
def softmax_cross_entropy(labels, logits):
    soft_logits = tf.nn.softmax(logits)
    soft_logits = tf.clip_by_value(soft_logits, 0.000001, 0.999999)
    cross_entropy = -tf.reduce_sum(labels * tf.log(soft_logits), axis=1)
    return cross_entropy
```

`tf.nn.softmax()`的作用为实现softmax操作的，但有一点值得要说的就是`.softmax(dim=-1)`中的参数`dim`。`dim`的默认值是-1，也就是默认将对最后一个维度进行softmax操作。例如在现在这个分类任务中，最后得到`logits`的维度为`[n_samples,n_class]`，并且我们也的确需要在最后一个维度（每一行）进行softmax操作，因此我们也就没有传入`dim`的值。可需要我们注意的是，**不是所有情况下我们需要进行softmax操作的维度都是最后一个，所有应该要根据实际情况通过`dim`进行指定**。

`tf.clip_by_value()`的作用是对传入值按范围进行裁剪。例如上面第三行代码的作用就是将`soft_logits`的值限定在`[0.000001,0.999999]`中，当小于或者大于边界值时，将会强制设定为对于边界值。

`tf.log()`表示取自然对数，即$y = \log_e x$。同时，倒数第二行代码则是用来计算所有样本的交叉熵损失。当然，这个`softmax_cross_entropy`这个函数的功能tensorflow也已经帮我们实现好了，在下一篇文章中我们会对其进行介绍。

### 3.3 定义模型

```python
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
    loss = tf.reduce_mean(cross_entropy)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

由于代码稍微有点长，为了排版美观就分成了定义模型和训练模型来进行介绍，其实两者都在一个函数中。上面代码中，前面大部分都是我们在上一篇文章中说过的，在这里就不再赘述。其中倒数第5、6行代码计算得到交叉熵损失值；`tf.train.AdamOptimizer()`为基于梯度下降算法改进的另外一种考虑到动量的优化器。

倒数第二行则用来计算预测正确和错误的样本，其中`tf.argmax()`的作用为取概率值最大所对应的索引（即类标）。例如`tf.argmax([0.2,0.5,0.3])`的结果为0.5所对应的下标，类似的还有`tf.argmin()`。`tf.equal()`则用来判断两个输入是否相等的情况，例如`tf.equal([0,5,3,6,1],[0,5,3,1,1])`的结果为`[True,True,True,False,True]`。

最后一行代码则是用来计算预测的准确率，其中`tf.cast()`表示进行类型转换，即可以将`True`转换为1，`False`转换为0，这样就可以通过计算平均值得到准确率。

### 3.4 训练模型

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(gen_batch(x_train, y_train, 					batch_size=batch_size)):
            batch_x = batch_x.reshape(len(batch_y), -1)
            feed_dict = {x: batch_x, y: batch_y}
            l, acc, _ = sess.run([loss, accuracy, train_op], feed_dict=feed_dict)
            if step % 100 == 0:
                print("Epochs{}/{}--Batches{}/{}--loss on train:{:.4}--acc: {:.4}".
                      format(epoch, epochs, step, len(x_train) // batch_size, l, acc))
        if epoch % 2 == 0:
            total_correct = 0
            for batch_x, batch_y in gen_batch(x_test, y_test, batch_size):
                batch_x = batch_x.reshape(len(batch_y), -1)
                feed_dict = {x: batch_x, y: batch_y}
                c = sess.run(correct_prediction, feed_dict=feed_dict)
                total_correct += np.sum(c * 1.)
                print("Epochs[{}/{}]---acc on test: {:.4}".
                      format(epoch, epochs, total_correct / len(y_test)))            
```

这部分代码仍旧是函数`train()`中的代码，可以发现里面大部分的知识点我们上一篇文章中已经做过介绍了。这里稍微需要说一下的就是在本示例中，每个epoch喂入的数据并不是全部样本，而是分batch进行输入。其次是在测试集上计算准确率时，我们先是得到了所有batch中预测正确的样本数量，然后除以总数得到准确率的。最后需要注意的是，在`sess.run()`中如果是计算多个节点的值，则传入的应该是一个list（例如上面代码第七行）；如果仅仅只是计算一个节点的值，则传入改节点即可（例如上面代码倒数第四行）。

### 3.5 运行结果

```python
Epochs[0/10]---Batches[0/234]---loss on train:3.253---acc: 0.07031
Epochs[0/10]---Batches[100/234]---loss on train:0.5581---acc: 0.7891
Epochs[0/10]---Batches[200/234]---loss on train:0.5026---acc: 0.8359
Epochs[0/10]---acc on test: 0.8149
Epochs[1/10]---Batches[0/234]---loss on train:0.4793---acc: 0.8555
Epochs[1/10]---Batches[100/234]---loss on train:0.4411---acc: 0.8281
Epochs[1/10]---Batches[200/234]---loss on train:0.444---acc: 0.8672
Epochs[2/10]---Batches[0/234]---loss on train:0.4289---acc: 0.8789
Epochs[2/10]---Batches[100/234]---loss on train:0.4132---acc: 0.8398
Epochs[2/10]---Batches[200/234]---loss on train:0.421---acc: 0.8672
Epochs[2/10]---acc on test: 0.8335
```

## 4 总结

在这篇文章中，笔者首先介绍了如何定义Softmax回归的正向传播以及如何计算得到交叉熵损失函数；然后介绍了如何定义模型中的参数以及准确率的计算等。并同时依次介绍了Tensorflow中所涉及到的各个API，例如`tf.nn.softmax()`，`tf.clip_by_value()`和`tf.equal()`等。

本次内容就到此结束，感谢您的阅读！若有任何疑问与建议，请添加笔者微信'nulls8'进行交流。青山不改，绿水长流，我们月来客栈见！

### 引用

[1]示例代码：https://github.com/moon-hotel/Tensorflow1.xTutorials



## [<返回主页>](../README.md)