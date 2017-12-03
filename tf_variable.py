import tensorflow as tf

state = tf.Variable(0, name='counter')
one = tf.constant(1)
new_value = tf.add(state, one)

update = tf.assign(state, new_value)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
sess.run(update)
print(sess.run(state))
