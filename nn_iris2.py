import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data

# 70 de entrenamiento
x_data_train = data[0:105, 0:4].astype('f4')
y_data_train = one_hot(data[0:105, 4].astype(int), 3)

# 15 de validacion
x_data_val = data[106:129, 0:4].astype('f4')
y_data_val = one_hot(data[106:129, 4].astype(int), 3)

# 15 de test
x_data_test = data[130:150, 0:4].astype('f4')
y_data_test = one_hot(data[130:150, 4].astype(int), 3)

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 20

for epoch in xrange(100):
    for jj in xrange(len(x_data_train) / batch_size):
        batch_xs = x_data_train[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_train[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    print "VALIDACION"
    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    result = sess.run(y, feed_dict={x: x_data_val})
    for b, r in zip(y_data_val, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"

print "TEST"

error = 0
result = sess.run(y, feed_dict={x: x_data_test})
i = 1
for b, r in zip(y_data_test , result):
    print i, "-->" ,b, "-->", r
    if np.argmax(b) != np.argmax(r):
		print "-----> ERROR"
		error = error + 1
    i += 1

print "Errores = ",error


