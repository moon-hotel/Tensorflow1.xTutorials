# Tensorflow1.xTutorials
<center>
    <img src="https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/0185.png" style="zoom:57%;" />
</center>

# 目录

## 第零讲：初识Tensorflow

- [简单谈谈Tensorflow的运行机制](01_LinearRgression/01_AboutTensorflow.md)

## 第一讲：线性回归

- [Tensorflow实现线性回归](01_LinearRgression/02_LinearRegression.md)

## 第二讲：Softmax回归

- [Tensorflow实现Softmax回归](02_SoftmaxRegression/01_SoftmaxRegression.md)
- [Tensorflow中两种交叉熵接口的用法](02_SoftmaxRegression/02_CrossEntropyUsage.md)

## 第三讲：深度前馈神经网络

- [Tensorflow实现深度前馈神经网络](03_DeepForwardNN/01_DeepForwardNN.md)

  



### [知识点索引](KnowledgeIndex.md)

## 1 前言

最近看到群里好几位同学都在吐槽Tensorflow比较难用，不对是相当难用（针对的是1.x版本）。其实说来也是，记得笔者当初在初学Tensorflow的时候同样也是一片茫然：例如什么是`Placeholder`？为什么每次定义变量的时候还要通过`name`来起一个名字？为什么每次运行代码的时候都要开始一个`session`？等等之类的问题。不过随着你慢慢了解到Tensorflow的相关知识后，你可能也会觉得这样设计确实有它的独到之处。为了尽可能的帮助更多的同学学习了解Tensorflow的使用方法，因此笔者将会在接下来的一个系列中以程序案例的方式来介绍Tensorflow1.x的使用方法。

## 2 Tensorflow1.x指南

### 2.1 关于学习路线

在接下来的一段时间，笔者将开设一个新的专辑**《Tensorflow入门指南》**来介绍Tensorflow1.x版本的使用方法。《Tensorflow入门指南》的主要介绍思路为以Tensorflow1.15为基础，通过以如何实现各个经典模型（例如线性回归、Softmax回归、CNN、RNN、BILSTM、CNN-LSTM等）为主干线，来对Tensorflow中的各类知识点进行介绍。

### 2.2 关于内容

由于《Tensorflow入门指南》这个系列的文章将是以介绍Tensorflow的使用方法为主，所以只会涉及到框架相关知识的介绍。因此，对于各类模型的算法原理并不会涉及到太多。关于深度学习入门的这部分内容将会放到下一个系列《跟我一起深度学习》中来进行介绍。

### 2.3 关于版本选择

有同学可能会问，那为什么要选择1.15这个版本呢？能不能选择其它的版本例如1.5呢？因为自己刚好装的就是1.5。答案是尽量选择1.15，因为用过Tensorflow的人都知道，不同版本间的差异性还是挺大的，表现形式就是这个版本能运行，换一个版本就报错了。其次就是现在Tensorflow也在开始大力推行2.0版本，而1.15是目前1.0时代的最后一个版本，在未来一段时间都不会发生变化，所以学会之后也不会太担心再次发生改变。

### 2.4 关于参考

用过Tensorflow的同学可能都知道，在Tensoflow中可能同一个功能有不同的实现方法；并且要命的是官方也还没个统一示例，各类实现方式就是五花八门的。所以，笔者在实现这些模型的时候也会先以官方给出的示例为主，如果没有再辅以github上start数较高的代码，并且同时也会给出相应的出处供大家参考。

### 2.5 关于更新

由于工作的原因，因此对于《Tensorflow入门指南》这个系列的文章更新频率肯定也就会慢下来。当然，笔者也会在保证文章质量的前提下，尽量保持着以一周推送两到三次的频率进行更新，以不负各位读者的期待。

本次内容就到此结束，感谢您的阅读！若有任何疑问与建议，请添加笔者微信'nulls8'进行交流。青山不改，绿水长流，我们月来客栈见！

<center>
    <img src="https://moonhotel.oss-cn-shanghai.aliyuncs.com/images/000000.png" style="zoom:70%;" />
</center>




### 