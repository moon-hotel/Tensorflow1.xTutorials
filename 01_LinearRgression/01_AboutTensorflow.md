## 1 前言

由于Tensorflow采用了全新的静态图设计模式，所以其运行机制与我们脑中所熟悉的动态图有着截然不同之处。TensorFlow翻译成中文就是张量流，所以TensorFlow至少代表着两个概念：“**张量**”和“**流**”。这儿我们不过多的追究什么是张量，在Tensorflow中它基本上就相当于numpy中的`array`，下面关键要说的是这个“流”。

怎么来说明这个“流”呢？我们先来看一段用python写的普通代码：

```python
a=1
print("a=",a) # a = 1
b=2
print("b=",b) # b = 2
c=a+b
print("c=",c) # c = 3
d=b+2
print("d=",d) # d = 4
e=c*d
print("e=",e) # e = 13
```

这看起来似乎也很平常没有什么特别之处，当然这确实没什么值得要说的。之所以这么认为是因为没有对比，所谓没有对比就没有伤害。下面我们用TensorFlow框架再来把这段程序写一遍：

```python
import tensorflow as tf
a=tf.constant([1],dtype=tf.int32,name='iama')
print(a)
b=tf.constant([2],dtype=tf.int32,name='iamb')
print(b)
c=a+b
print(c)
d=b+2
print(d)
e=c*d
print(e)

结果：
Tensor("iama:0", shape=(1,), dtype=int32)
Tensor("iamb:0", shape=(1,), dtype=int32)
Tensor("add:0", shape=(1,), dtype=int32)
Tensor("add_1:0", shape=(1,), dtype=int32)
Tensor("mul:0", shape=(1,), dtype=int32)
```

发现没有，居然和我们想象中的结果不一样，输出来的只是每个变量的信息。那这么样才能得到我们想象中的结果呢？在第18号后面加上如下几句代码即可：

```python
with tf.Session() as sess:
    print(sess.run(a))
    print(sess.run(b))
    print(sess.run(c))
    print(sess.run(d))
    print(sess.run(e))
```

为什么Tensorflow需要通过这种形式来执行代码呢？

## 2 计算图

从上面的两个例子可以发现：

（1）传统方式写的程序，对于任意一个变量，我们随时都可以输出其结果，这种称为动态图模式；

（2）用TensorFlow框架写的程序，对于每一个变量，我们必须`run()`一下才能得到其对应的结果。

在`Tensorflow`中为什么会这样呢？根据上面的程序，我们可以大致画出如下一个计算图：

<img src="https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000139.png" style="zoom:80%;" />

<center>
    图 1. 计算图
</center>



如图1所示，里面的**每一个节点**表示的仅仅只是其对应的结构信息（形状，类型，操作等），并不是代表其真实的值。如果直接输出节点，实际上则是输出的对应节点的信息。例如a节点为`"iama:0", shape=(1,), dtype=int32`；c节点为Tensor("add:0", shape=(1,), dtype=int32)`，其中`add`是指c这个变量是通过加法操作得到的。

而**每一条边**则表示各个变量之间的依赖关系。如c依赖于a,b；a,b不依赖于任何变量。当执行`run()`的时候，就对应执行每个节点上所对应的操作，并且返回这个操作后的结果，这就是TensorFlow的设计理念——**先定义计算图，等计算图定义完成后再通过会话模式来执行计算图上的操作**。

例如：

在执行`run(a)`时，就对应把1赋值给常量a;

在执行`run(c)`时，就对应先把1赋值给常量a，再把2赋值给常量b,最后把a+b这个操作后的结果赋值给c；

在执行`run(e)`时，就对应把图中所有的操作都执行一遍，也就是说TensorFlow会根据图中的依赖关系自动处理。

因此，对于上述运行机制，我们可以通过图2来形象的进行表示。可以发现，所有的操作就像是水一样从初始节点“流”向终止节点。

![](https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000140.gif)

<center>
    图 2. Tensorflow运行机制图
</center>



图片来自：https://dorianbrown.dev/mnist-tensorflow/

由此我们可以知道：`run(node)`的本质就是执行所有node所依赖的节点对应的操作，并且返回node节点对应操作后的值。所以，利用TensorFlow框架写程序时定义的每一变量、常量，其实都只是在计算图上生成了一个对应的节点。

## 3 占位符

说完计算图我们再来说Tensorflow中的占位符（`tf.placeholder`）就很容易理解了。从图1可知，c、d和e三个节点是依赖于其前驱节点所生成；而唯独节点a和b并不依赖于任何前驱节点。这是因为节点a和b是计算图中的原始输入，所以才不会依赖于其它节点。但是虽然说a和b不依赖于别的节点，但是别的节点得依赖于a和b呀。怎么办呢？

Tensoflow说那既然是这样，咱们就**先给它挖个坑**把地方占住吧，等计算图执行的时候我们再给这些坑填入原始的输入即可。就这样，Tensorflow中诞生了一个让很多人莫名疑惑的`tf.placeholder()`。

本次内容就到此结束，感谢您的阅读！若有任何疑问与建议，请添加笔者微信'nulls8'进行交流。青山不改，绿水长流，我们月来客栈见！



### [<返回主页>](../README.md)