import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


# 从[-1,1]中均匀地取300个数组成数组，[:,np.newaxix]表示增加一个维度,:表示当前数组的全部
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
# 生成和x_data一样结构的高斯分布,平均值为0,方差为0.05
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise
