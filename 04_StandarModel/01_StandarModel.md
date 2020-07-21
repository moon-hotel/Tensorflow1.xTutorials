## 1 前言

在前面几篇文章中，笔者已经陆续介绍了如何通过Tensorflow来实现**[线性回归](../01_LinearRgression/02_LinearRegression.md)**、**[Softmax回归](../02_SoftmaxRegression/01_SoftmaxRegression.md)**以及**[深度前馈神经网络](../03_DeepForwardNN/01_DeepForwardNN.md)**等。但是在实现这些模型时，为了忽略Python语法等额外知识点对API理解造成的影响，因此笔者在书写这些代码的时候往往都是通过定义多个函数来实现。这种做法算法也能使模型跑起来，但总显得不那么美观，更重要的是无法便捷的修改模型中的参数。同时，到目前为止我们还没有介绍如何对训练过程中模型的参数进行保存，以及如何再次载入本地模型进行训练等。

因此在这篇文章中，笔者仍旧以实现一个深度前馈神经网络为例，但会将其抽象成一个类的形式。同时，也会介绍Tensorflow的模型保存和加载机制等。

## 2 类的定义

### 2.1 定义初始化方法

```python
# model.py

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
```

在实现一个类的时候，我们可能一开始并不会知道哪些参数需要初始化，不过没关系用到一个再来`__init__()`里面定义就行。这个类的初始化方法在实例化一个类（`model = DeepNN()`）对象时就会被调用执行。

### 2.2 定义占位符

```python
 def _build_placeholder(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.input_nodes], 			name='input_x')
        self.y = tf.placeholder(dtype=tf.int64, shape=[None], name="input_y")
```

这是这个类需要实现的第二个方法，定义好模型需要用到的占位符。由于`x,y`需要在后面方法中用到，所以将其定义为了类的成员变量。

### 2.3 定义网络模型

```python
def _build_net(self):
    with tf.name_scope('forward'):
        fc = tf.layers.dense(self.x, units=self.hidden_1_nodes)
        fc = tf.layers.dense(fc, units=self.hidden_2_nodes, activation=tf.nn.relu)
        fc = tf.layers.dense(fc, units=self.hidden_3_nodes, activation=tf.nn.relu)
        self.logits = tf.layers.dense(fc, units=self.output_nodes)
	with tf.name_scope('net_loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_w(self.y,self.logits)
        self.loss = tf.reduce_mean(cross_entropy) + get_regularization_loss()
	with tf.name_scope('accuracy'):
        self.correct_prediction = tf.equal(tf.argmax(self.logits,1), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction))

```

注意，为例排版美观上述代码部分函数名并不完整（参见示例代码）。从上面的示例代码可以看到，其与之前写法在逻辑上并没有太大的区别。只是这里多了`tf.name_scope()`这么个东西，它有什么用呢？简单来说它就像是一个围栏一样[1]，你可以把由某几行代码组成一个小的功能块放到一个`scope()`中，这样做的好处：一是可以使得代码的逻辑更加清晰；二是在用Tensorboard对网络结构进行可视化时，同一个`scope()`下的语句会被看出是一个功能块进行折叠，这样会更加有利于查看模型的整体结构。等到后面介绍Tensorboard时再来细说。

### 2.4 定义网络训练

由于这部分代码稍微有点多，因此下面只列出了之前没有介绍过的部分，完整部分可参见文末代码示例。

```python
def train(self, x_train, y_train, x_test, y_test, gen_batch):
    if not os.path.exists(self.model_path):
        os.makedirs(self.model_path)
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=3)
    model_name = "model"
    with tf.Session() as sess:
        start_epoch = 0
        check_point = tf.train.latest_checkpoint(self.model_path)
        if check_point:
            saver.restore(sess, check_point)
            start_epoch += int(check_point.split('-')[-1])
            print("### Loading exist model <{}> successfully...".format(check_point))
        for epoch in range(start_epoch, self.epochs):
            .....
            if epoch % 2 == 0:
                acc = self.evaluate(sess, x_test, y_test, gen_batch)
                saver.save(sess, os.path.join(self.model_path, model_name + 		                   "_{}".format(acc)),global_step=epoch, write_meta_graph=False)
                print("### Saving  model successfully...")
                print("Epochs{}/{}--acc on test:{}".format(epoch, self.epochs, acc))
```

上述代码的第2，3行用来判断保存模型的文件夹是否存在，不存在则创建一个。第4行代码则是定义一个用于模型保存的实例化对象，`tf.global_variables()`表示获得全局的变量以便保存，`max_to_keep=3`表示只保存最近最新的三个模型。第8行代码用来查找指定路径下的模型参数，如果存在则第10行代码用于载入模型。倒数第5行代码用于每两个epoch对测试集的准确率进行测试，同时倒数第4行代码将对此时的模型参数进行保存。可以发现，模型的保存的文件名中我们还加入了此时所对应的在测试集上的准确率。如下图1所示便为模型保存后的结果图，其中`.data-`这个文件中保存的就是模型的权重参数。

<img src="https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000142.png" style="zoom:80%;" />

<center>
    图1. 网络模型保存图
</center>


### 2.5 定义评估函数

```python
def evaluate(self, sess, x_test, y_test, gen_batch):
    total_correct = 0
    for batch_x, batch_y in gen_batch(x_test, y_test, self.batch_size):
        batch_x = batch_x.reshape(len(batch_y), -1)
        feed_dict = {self.x: batch_x, self.y: batch_y}
        c = sess.run(self.correct_prediction, feed_dict=feed_dict)
        total_correct += np.sum(c * 1.)
    acc = total_correct / len(y_test)
    return acc
```

在这里，我们专门把在测试集上的准确率计算过程抽象成了一个函数，这样方便在需要的地方进行调用。

## 3 模型训练

上述的整个模型的定义都是放在`model.py`这个文件中完成的，下面我们将在另一个文件`train.py`中来利用我们写好的类进行模型的实例化，然后训练模型。

### 3.1 导入类模型和数据集

```python
from tensorflow.keras.datasets.fashion_mnist import load_data
from model import DeepNN

def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = load_data()
    return x_train / 255., y_train, x_test / 255., y_test
```

上述第2行代码就是将我们定义好的类`DeepNN`导入进来，以便于后续使用。

### 3.2 开始训练模型

```python
if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_mnist_data()
    model = DeepNN(learning_rate=0.001,
                   epochs=10,
                   batch_size=256,
                   input_nodes=784,
                   hidden_1_nodes=1024,
                   hidden_2_nodes=1024,
                   hidden_3_nodes=256,
                   output_nodes=10,
                   regularization_rate=0.0001,
                   model_path='MODEL')
    model.train(x_train, y_train, x_test, y_test, gen_batch)
```

首先我们需要通过第2行代码载入数据集；然后开始实例化一个`DeepNN()`的对象`model`；最后表示通过这个`model`对象来调用模型的训练方法`.train()`。

### 3.3 结果

```python
Epochs[2/10]---Batches[200/234]---loss on train:0.4229---acc: 0.8984
### Saving  model successfully...
Epochs[2/10]---acc on test: 0.8634
Epochs[3/10]---Batches[0/234]---loss on train:0.3666---acc: 0.918
Epochs[3/10]---Batches[50/234]---loss on train:0.3412---acc: 0.9023
Epochs[3/10]---Batches[100/234]---loss on train:0.3987---acc: 0.9023
Epochs[3/10]---Batches[150/234]---loss on train:0.3627---acc: 0.9102
Epochs[3/10]---Batches[200/234]---loss on train:0.4013---acc: 0.9062
Epochs[4/10]---Batches[0/234]---loss on train:0.3727---acc: 0.9141
Epochs[4/10]---Batches[50/234]---loss on train:0.3442---acc: 0.9141
Epochs[4/10]---Batches[100/234]---loss on train:0.3662---acc: 0.8945
Epochs[4/10]---Batches[150/234]---loss on train:0.349---acc: 0.9141
Epochs[4/10]---Batches[200/234]---loss on train:0.3621---acc: 0.9102
### Saving  model successfully...
Epochs[4/10]---acc on test: 0.8672
```

## 4 总结

在这篇文章中，笔者首先将上一篇文章中的代码进行了重构，将其抽象成了一个类的形式；其次还介绍了在Tensorflow中如何对训练好的模型进行保存，以及如何载入已有的模型进行追加训练。

本次内容就到此结束，感谢您的阅读！如果你觉得上述内容对你有所帮助，欢迎关注并传播本公众号！若有任何疑问与建议，请添加笔者微信'nulls8'进行交流。青山不改，绿水长流，我们月来客栈见！

### 引用

[1]`tf.name_scope()`作用：https://www.jianshu.com/p/635d95b34e14

[2]示例代码：https://github.com/moon-hotel/Tensorflow1.xTutorials





## [<返回主页>](../README.md)