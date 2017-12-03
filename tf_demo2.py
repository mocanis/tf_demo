import tensorflow as tf
import numpy as np

x = np.random.rand(100).astype(np.float32)
y = x * 0.1 + 0.3

W = tf.Variable(tf.random_uniform([1], -1.0, 0))
b = tf.Variable(tf.zeros([1]))

y_hat = W * x + b
loss = tf.reduce_mean(tf.square(y_hat - y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)

for step in range(201):
    session.run(train)
    if step % 20 == 0:
        print(step, session.run(W), session.run(b))
