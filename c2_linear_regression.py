import tensorflow as tf
import numpy as np

# X = [1, 2, 3, 4] <=> [x0, x1, x2, x3]
# Y = [2]  <=>  w_init + w0*x0 + w1*x1 + w2*x2 + w3*x3

# Fake x_datas, sample_weights, y_datas
x_datas = np.random.random((10000,2))
sample_weights = np.array([3, 4]).reshape(2, )  # gia tri W chinh xac de tinh Y theo X la [3,4]
y_datas = np.matmul(x_datas, sample_weights)
y_datas = np.add(y_datas, np.random.uniform(-0.5, 0.5))
y_datas = y_datas.reshape(len(y_datas), 1)

# tach tap train va tap test
from sklearn.model_selection import train_test_split
x_trains, x_tests, y_trains, y_tests = train_test_split(x_datas, y_datas, test_size=0.2, random_state=42)
n_dim = x_trains.shape[1]; # chieu cua vector train

# placeholder de truyen du lieu
X = tf.placeholder(tf.float32, [None, n_dim])
Y = tf.placeholder(tf.float32, [None, 1])

# variable dung de training
W = tf.Variable(tf.ones([n_dim, 1]))  # gia tri W ban dau la [1,1]
b = tf.Variable(np.random.randn(), dtype=tf.float32)  # gia tri random

pred = tf.add(tf.matmul(X, W), b)           # tinh gia tri thuc te voi gia tri W, b ngau nhien ban dau
loss = tf.reduce_mean(tf.square(pred - Y))  # giam thieu sai sot voi pred la gia tri thuc te, Y la gia tri dung
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)   # dung phuong phap hoi quy tuyen tinh
sess = tf.Session()
init = tf.global_variables_initializer().run(session=sess)

epochs = 10000  # so lan training, training cang nhieu se cang giam gia tri loss, ket qua cang chinh xac
for epoch in range(epochs):
    sess.run(optimizer, feed_dict={X : x_trains, Y : y_trains})   # training de tinh lai W, loss
    test_loss = sess.run(loss, feed_dict={X : x_tests, Y : y_tests})   # dung tap test de kiem tra xem gia tri loss giam dan sau moi lan training
    if epoch % 5000 == 0:
        print('Epoch {} - Test lost = {}'.format(epoch, test_loss))
        print(sess.run(W))
print('Training finished !!!')
print(sess.run(W))  # gia tri W sau khi training

# kiem tra xem chuong trinh du doan nhu the nao
pred_y = sess.run(pred, feed_dict={X : [[5,5]]})
print(pred_y)
    
