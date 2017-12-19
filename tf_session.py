import tensorflow as tf

matrix1 = tf.constant([[3, 3]])#1行2列
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)
print(product)
session = tf.Session()
result = session.run(product)
print(result)
