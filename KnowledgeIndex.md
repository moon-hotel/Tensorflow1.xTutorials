# API知识点索引目录

## 01-线性回归

- [Tensorflow实现线性回归](01_LinearRgression/02_LinearRegression.md)

  ​	**关键字：Tensorflow安装、内置数据集、MSE损失函数**

  - `tensorflow.keras.datasets`
  - `tf.matmul()`
  - `tf.nn.xw_plus_b()`
  - `tf.reduce_mean()`
  - `tf.square()`
  - `tf.reduce_mean()`
  - `tf.placeholder()`
  - `tf.Variable()`
  - `tf.truncated_normal()`
  - `tf.constant()`
  - `GradientDescentOptimizer()`
  - `tf.global_variables_initializer()`

## 02-Softmax回归

- [Tensorflow实现Softmax回归](02_SoftmaxRegression/01_SoftmaxRegression.md)
  
  ​	**关键字：分类模型、交叉熵、准确率、分类损失函数**
  
  - `tf.nn.softmax()`
  - `tf.clip_by_value()`
  - `tf.log()`
  - `tf.train.AdamOptimizer()`
  - `tf.argmax(),tf.argmin()`
  - `tf.equal(),tf.cast()`
  
- [Tensorflow中两种交叉熵接口的用法](02_SoftmaxRegression/02_CrossEntropyUsage.md)

  **关键字：softmax、交叉熵**

  - `softmax_cross_entropy_with_logits_v2`
  - `sparse_softmax_cross_entropy_with_logit`

## 03-深度前馈神经网络

- [Tensorflow实现深度前馈神经网络](03_DeepForwardNN/01_DeepForwardNN.md)

  ​	**关键字：全连接层、正则化**

  - `tf.layers.dense()`

  - `tf.nn.relu`、`tf.sigmoid`、`tf.tanh`

  - `tf.constant_initializer`

  - `tf.glorot_normal_initializer`

  - `tf.glorot_uniform_initializer`

  - `tf.truncated_normal_initializer`

  - `tf.keras.regularizers.l2(0.1)`

  - `tf.keras.regularizers.l1(0.1)`

  - `tf.keras.regularizers.l1_l2(l1=0.1, l2=0.2)`

  - `tf.losses.get_regularization_loss()`

  - `tf.losses.get_regularization_losses()`

## 04-完整的网络模型示例

- [一个完整的网络模型示例](04_StandarModel/01_StandarModel.md)

  **关键词：模型保存、模型加载**

  - `tf.name_scope()`
  - `tf.train.Saver()`

- [利用Tensorboard进行可视化](04_StandarModel/02_Tensorboard.md)

  **关键词：Tensorboard、可视化**

  - `tf.name_scope()`
  - ` tf.summary.FileWriter()`
  - `tf.variable_scope()`
  - ` tf.get_varibale()`
  - `tf.summary.scalar()`
  - `tf.summary.histogram()`
  - `tf.summary.merge_all()`
  - `.add_summary()`





## [<返回主页>](README.md)