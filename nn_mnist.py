import gzip
import cPickle
import sys

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


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_x, train_y = train_set
train_y=one_hot(train_y.astype(int),10)
val_x, val_y = valid_set
val_y=one_hot(val_y.astype(int),10)
test_x,test_y = test_set
test_y=one_hot(test_y.astype(int),10)



# TODO: the neural net!!
x = tf.placeholder("float", [None, 784])  # samples
y_ = tf.placeholder("float", [None, 10])  # labels

W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print "----------------------"
print "   Start training...  "
print "----------------------"

batch_size = 1000
last = sys.maxint
now = 0
epoch = 0
while now > 0.001*last +last or now < last - 0.001*last:
    last = now
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
    #print "VALIDACION"
    now = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    print "Epoch #:", epoch, "Error: ", now
    epoch += 1

    #result = sess.run(y, feed_dict={x: val_x})
    #for b, r in zip(val_y, result):
    #    print b, "-->", r
    #print "----------------------------------------------------------------------------------"

print "TEST"

error = 0
result = sess.run(y, feed_dict={x: test_x})
i = 1
for b, r in zip(test_y , result):
    #print i, "-->" ,b, "-->", r
    if np.argmax(b) != np.argmax(r):
		#print "-----> ERROR"
		error = error + 1
    i += 1

print "Errores = ",error