import tensorflow as tf
print(tf.__version__)

# Constant  =  Khong thay doi trong toan bo session
x1 = tf.constant(1, name='cons_x1', dtype='int32')
x2 = tf.constant(2, name='cons_x2', dtype='float32')
v1 = tf.constant([[4,5,6],[1,2,3]], name = 'cons_v1', dtype=tf.float32)
v2 = tf.constant([[1,2,3],[4,5,6]], name = 'cons_v2', dtype=tf.float32)
sess = tf.Session()
v3 = v1 + v2
print(sess.run(v3))

# Placeholder  =  Load gia tri dau vao co san
p1 = tf.placeholder(dtype=tf.float32)
p2 = tf.placeholder(dtype=tf.float32)
o_add = p1 + p2
o_multi = p1 * p2
o_delta = p1**2 + p2

d_values = {
    p1 : 20,
    p2 : 10
}
print(sess.run(o_add, feed_dict=d_values))
print(sess.run([o_add, o_multi, o_delta], feed_dict=d_values))

# Variable  =  Bien thay doi trong chuong trinh
var1 = tf.Variable(name='var1', initial_value=20)
var2 = tf.Variable(name='var2', initial_value=30)
tf.global_variables_initializer().run(session=sess)
print(sess.run(var1 + var2))

sess.close()