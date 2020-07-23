

## 1 前言

在上一篇文章中，笔者介绍了如何实现一个较为规范和完整的网络模型，包括如何将模型抽象成一个类、如何实现模型的保存和载入等。在接下来的这篇文章中，笔者将会介绍Tensorflow中一个有利的可视化工具Tensorboard。这个包一般在安装Tensorflow的一会自动的被安装上，所以我们并不需要额外的对其进行安装。同时，为了不被其它信息所干扰，在下面介绍Tensorboard的过程中，笔者仍旧以一段面向过程代码进行介绍。

## 2 Tensorboard可视化

说到Tensorboard的可视化，其主要囊括了两个方面的可视化：网络结构可视化以及参数（变量）的可视化。下面首先就来说说网络结构的可视化。

### 2.1 网络结构可视化

```python
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
            fc = tf.layers.dense(x, units=1024, activation=tf.nn.relu)
        with tf.name_scope('fc2'):
            logits = tf.layers.dense(fc, units=num_class)
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.sparse_softmax_cross_(labels=y,logits=logits)
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(cross_entropy)
    with tf.name_scope('train_op'):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(logits, axis=1), y)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.summary.FileWriter('logs/', sess.graph)

```

如上所示为一段仅包含两个全连接层的分类模型代码，并且里面还刻意的加了很多`tf.name_scope`语句。想要对这个网络结构进行可视化的关键在于最后一行代码，他表示将生成的这个网络图写入到本地的`logs`文件夹（如图1所示）中（当然名字可以任取）。最后，我们只需要通过Tensorboard这个工具来打开生成的这个文件即可。

![](https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000143.png)

<center>
    图1. logs文件夹
</center>


- 步骤一

  首先第一步是打开终端（windows CMD命令行），然后激活一个装有Tensorboard的虚拟环境（不然后面会报错找不到tensorboard这个命令），同时进入到与`logs`文件夹同级的目录：

  ![](https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000144.png)

  <center>
      图2. logs文件夹同级目录
  </center>

- 步骤二

  执行命令`tensorboard --logdir=logs`启动tensorboard。如果运行成功的话会出现如图3所示的结果：

  ![](https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000145.png)

  <center>
      图3. tensorboard运行成功图
  </center>

- 步骤三

  最后打开浏览器（推荐谷歌），输入图3中红色方框中的地址（具体视个人情况）打开即可。这样，我们就能看到如下所示的一张结构图：

  <img src="https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000146.png" style="zoom:50%;" />

  <center>
      图4. tensorboard网络结构图 
  </center>


  从图4中可以到，每一个灰色圆角矩形对应的都是代码中的一个`tf.name_scope()`块儿，并且各个块儿之间的依赖关系都可以通过对应的灰色箭头进行追溯。同时，如果我们再双击layer这个圆角矩形还能得到如图5所示的两个块儿fc1和fc2。因为在上面的代码中，我们也特意用`tf.name_scope()`这个语句分别对两个全连接层进行了包裹。

  <img src="https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000147.png" style="zoom:40%;" />

<center>
    图5. name scope作用图
</center>


到这里，对于如何通过tensorboard来对网络模型进行可视化就介绍完了。但在实际情况中，这项功能用得都比较少，用得比较频繁的应该是对网络中的参数进行可视化。

### 2.2 参数可视化

可视化网络计算图不是太有意义，而更有意义的是在训练网络的同时能够看到一些参数的变换曲线图（如：准确率，损失等），以便于更好的分析网络。要实现这个操作，只需要添加对应的`tf.summary.scalar('acc', acc)`语句，然后最后合并所有的`summary`即可。但是，通常情况下网络层的权重参数都不是标量，因此通常的做法就是计算其最大、最小、平均值以及直方图等（可以将自己需要的进行添加）。由于对于很多参数都会用到同样的这几个操作，所以在这里就统一定义函数：

```python
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
```

最后在需要可视化参数的地方，调用这个函数即可。

但现在有一个问题就是，对于封装好的API如`tf.layers.dense()`等，我们应该如何取出其中的权重参数呢？解决办法就是通过变量名来获取：

```python
fc = tf.layers.dense(x, units=1024, activation=tf.nn.relu, name='fc1')
with tf.variable_scope("fc1", reuse=True):
	w1 = tf.get_variable("kernel")
    b1 = tf.get_varibale('bias')
```

通过上述几行代码，我们便能够获得这个`dense`层的权重参数。其中第2行代码的作用就是指明我们要获取的两个变量所在的变量域（范围），因为所有`dense`层权重参数的名字都相同；而`reuse=True`则表示该变量已经存在，我们仅作获取。值得一提的是，在某些网络中我们可能需要共享一些参数，而实现这一功能就需要结合`tf.variable_scope()`与`tf.get_variable()`，我们后续再做介绍。

如下代码就展示了如何实现对我们需要可视化的变量进行可视化，由于篇幅所限仅列出了关键代码，完整参见示例代码：

```python
def train(x_train, y_train, x_test, y_test):
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

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', loss)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('logs/', sess.graph)
        for epoch in range(epochs):
            for step, (batch_x, batch_y)
                l, acc, _, summary = sess.run([loss, accuracy, train_op, merged])
                writer.add_summary(summary,epoch*(len(x_train) // batch_size) + step)
```

对于需要可视化的权重，我们只需要调用`variable_summaries()`函数即可。同时，对于损失值等可以直接调用`tf.summary.scalar()`对其变化趋势进行可视化。在调用结束后，我们还需要再通过`tf.summary.merge_all()`来合并所有的`summary`；最后再通过` writer.add_summary()`将计算图每次运行得到的结果写入到本地即可。

同可视化网络结构图一样，我们仍旧要通过命令`tensorboard --logdir=logs`来对写入本地的数据在网页上进行展示。如图6所示，就是两个全连接层权重参数的可视化情况:

<img src="https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000148.png" style="zoom:70%;" />

<center>
    图6. 权重参数可视化图
</center>


如图7所示为准确率和损失值的变化情况，其中横坐标表示batch值。

<img src="https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000149.png" style="zoom:80%;" />

<center>
    图7. 准确率与损失可视化图
</center>


如图8所示为两个权重参数的直方图可视化情况：

<img src="https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000150.png" style="zoom:70%;" />

<center>
    图8. 权重参数直方图可视化
</center>


## 3 总结

在这篇文章中，笔者首先介绍了如何利用tensorboard对网络结构图进行可视化；然后进一步介绍了如何对模型中的参数进行可视化，并且顺便介绍了如何通过`tf.variable_scope()`和` tf.get_varibale()`来获取一个已经存在的变量。

本次内容就到此结束，感谢您的阅读！如果你觉得上述内容对你有所帮助，欢迎关注并传播本公众号！若有任何疑问与建议，请添加笔者微信'nulls8'进行交流。青山不改，绿水长流，我们月来客栈见！

### 引用

[1]权重获取：https://stackoverflow.com/questions/45372291/how-to-get-weights-in-tf-layers-dense

[2]示例代码：https://github.com/moon-hotel/Tensorflow1.xTutorials





