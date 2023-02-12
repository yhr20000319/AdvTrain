import tensorflow as tf2
tf = tf2.compat.v1
tf.disable_v2_behavior()

x2 = tf.constant([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])  # 4*3
y2 = tf.constant([1.0, 1, 2])  # 1*3
z2 = tf.multiply(x2, y2) # 等价于 z2= x2*y2
print("列元素一直自动复制行维度与相乘矩阵保持一致:", z2)

y3 = tf.constant([[1.0], [1], [2], [3]])  # 4*1
z3 = tf.multiply(x2, y3)# 等价于 z3 = x2 * y3
print("行元素一直自动复制列维度与相乘矩阵保持一致:", z3)

sess = tf.InteractiveSession()
print(z2.eval())
print(z3.eval())
