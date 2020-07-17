## 1 前言

在前面两篇文章中，笔者分别介绍了如何用Tensorflow来实现**线性回归**和**Softmax回归**，并且这两个模型有一个共同点就是均为单层的神经网络。那我们应该如何通过Tensorflow来实现一个多层的神经网络呢？有朋友可能就会说了，会写单层的难道还不会写多层了？确实，按照先前的做法：首先定义权重和偏置，然后完成矩阵乘法实现一个全连接层操作；接着再定义权重和偏置，完成第二个全连接层操作。可问题是，这样写一两层还好，可万一要写个十层八层的还不得累趴下？

可能还有的朋友会说到，自己定义一个全连接层，然后再调用即可。这确实不失为一个好的方法，不过恰好这一步工作Tensorflow已经帮我们做了，我们直接拿来使用即可。在接下来的这篇文章中，笔者就首先介绍一下在Tensorflow中全连接层` tf.layers.dense()`的使用方法；然后再来介绍如何对权重参数实现正则化的操作；最后再介绍如何通过它来实现一个多层的神经网络。

## 2 全连接层介绍

### 2.1 `.dense()`接口

```python
tf.layers.dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
```

如上所示为`tf.layers.dense()`函数在使用时所需要传入的参数，下面我们就来对一些常用的参数进行介绍：

- `inputs`：这个不用多说，输入张量；
- `units`：输出节点数，也就是说经过这层全连接后，输入张量的维度会变成`units`；
- `activation`：激活函数，其取值可以是如下
  - `tf.nn.relu`、`tf.sigmoid`、`tf.tanh`等；
- `use_bias`：是否使用偏置，默认为是；
- `kernel_initializer`：权重初始化方式，其取值可以是如下
  - `tf.constant_initializer`
  - `tf.glorot_normal_initializer`
  - `tf.glorot_uniform_initializer`
  - `tf.truncated_normal_initializer`；
- `kernel_regularizer`：权重的正则化方式，其取值可以是如下
  - `tf.keras.regularizers.l2(0.1)`，0.1表示惩罚系数
  - `tf.keras.regularizers.l1(0.1)`
  - `tf.keras.regularizers.l1_l2(l1=0.1, l2=0.2)`;

### 2.2 `.dense`使用示例

```python
import tensorflow as tf
x = tf.Variable(tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32))
weight = np.array([1, 1, 1, 0, 2, 1.]).reshape(3, 2)
fc = tf.layers.dense(x, units=2, activation=tf.nn.relu,
                     kernel_initializer=tf.constant_initializer(value=weight),
                     kernel_regularizer=tf.keras.regularizers.l2(0.1))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    f = sess.run(fc)
    print(f)
#结果
[[ 9.  4.]
 [21. 10.]]
```

如上代码所示为实现一个全连接层的方法，同时为了对其结果进行验证在这里我们使用了常数初始化方法来初始化内部的权重矩阵。也就是说此处的`tf.layers.dense()`完成了如下的一个矩阵乘法操作：
$$
fc = x\times weight=\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ \end{bmatrix} \times \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 2 & 1\\ \end{bmatrix}=\begin{bmatrix} 9 & 4 \\ 21 & 10 \\ \end{bmatrix} \tag{1}
$$

### 2.3 实现正则化

从上面的示例代码中可以看到，其实我们已经对权重参数`weight`施加了一`l2`个正则化。同时，如果按照预期的话，权重正则化后的应该是：
$$
reg = 0.1\times(1^2+1^2+1^2+0^2+2^2+1^2)=0.1\times 8=0.8\tag{2}
$$
但在Tensorflow中这个结果我们应该如何得到呢？在Tensorflow中，我们可以通过这么一个方法来获得施加正则化后的结果：`tf.losses.get_regularization_loss()`。

```python
l = tf.losses.get_regularization_loss()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    f, loss = sess.run([fc, l])
    print(f)
    print(loss)

#结果
[[ 9.  4.]
 [21. 10.]]
0.8
```

同时，如果稍加注意的话你会发现Tensorflow还提供了`tf.losses.get_regularization_losses()`这么一个方法。那这个方法又来干啥的呢？在一个复杂的网络中，如果你对很多参数都进行了正则化，那么这个复数形式的方法返回的就是一个列表，列表中的每个元素均为某个权重参数正则化后的结果，例如上述代码采用这个方法返回的结果则为`[0.8]`。而不带复数形式的方法返回的就是所有权重参数正则化后的累加和。

## 3 深度前馈神经网络

在这一小节中，我们将会实现一个简单的四层神经网络，并且用到的数据集依旧是**上一篇文章**中的fashion_mnist数据集，因此对于数据处理部分的介绍就不再赘述。

### 3.1 定义正向传播

```python
def forward(x, num_class):
    fc = tf.layers.dense(x, units=1024, activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    fc = tf.layers.dense(fc, units=1024, activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    fc = tf.layers.dense(fc, units=256, activation=tf.nn.relu,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    fc = tf.layers.dense(fc, units=num_class,
                         kernel_regularizer=tf.keras.regularizers.l2(0.0001))
    return fc
```

上述代码就完成了整个网络模型的构建，其中`x`为原始的输入，然后再将处理后的结果喂入到下一个全连接层即可。

### 3.2 定义训练模型

```python
def train(x_train, y_train, x_test, y_test):
    learning_rate = 0.005
    epochs = 10
    batch_size = 256
    n = 784
    num_class = 10
    x = tf.placeholder(dtype=tf.float32, shape=[None, n], name='input_x')
    y = tf.placeholder(dtype=tf.int64, shape=[None], name="input_y")
    logits = forward(x, num_class)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, 	           logits=logits)
    loss = tf.reduce_mean(cross_entropy) + tf.losses.get_regularization_loss()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, axis=1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

这段代码与上一篇文章中所介绍的总体上并没有什么差异，仅仅是加入了Tensorflow内置交叉熵的接口以及将正则化后的结果并入到损失函数中。

### 3.3 运行结果

```python
Epochs[1/10]---Batches[0/234]---loss on train:0.4778---acc: 0.8906
Epochs[1/10]---Batches[50/234]---loss on train:0.4625---acc: 0.8555
Epochs[1/10]---Batches[100/234]---loss on train:0.5046---acc: 0.8594
Epochs[1/10]---Batches[150/234]---loss on train:0.4584---acc: 0.8828
Epochs[1/10]---Batches[200/234]---loss on train:0.4518---acc: 0.8711
Epochs[2/10]---Batches[0/234]---loss on train:0.3928---acc: 0.8945
Epochs[2/10]---Batches[50/234]---loss on train:0.3618---acc: 0.8867
Epochs[2/10]---Batches[100/234]---loss on train:0.4225---acc: 0.9023
Epochs[2/10]---Batches[150/234]---loss on train:0.3992---acc: 0.8789
Epochs[2/10]---Batches[200/234]---loss on train:0.3964---acc: 0.8906
Epochs[2/10]---acc on test: 0.8567
```

## 4 总结

在这篇文章中，笔者首先介绍了Tensorflow中全连接层`tf.layers.dense()`的函数定义，并介绍了其中各个参数的作用和取值；然后介绍了如何通过`tf.layers.dense()`来实现对权重参数的正则化处理；接着再介绍了如何通过`get_regularization_loss()`将正则化后的权重值并入到分类损失函数中；最后介绍了如何通过`tf.layers.dense()`来实现一个简单的四层神经网络的分类任务。

本次内容就到此结束，感谢您的阅读！若有任何疑问与建议，请添加笔者微信'nulls8'进行交流。青山不改，绿水长流，我们月来客栈见！

### 引用

[1]示例代码：https://github.com/moon-hotel/Tensorflow1.xTutorials

[2]https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/layers/dense

[3]https://www.jianshu.com/p/3855908b4c29

[4]https://www.zhihu.com/question/275811771





## [<返回主页>](../README.md)