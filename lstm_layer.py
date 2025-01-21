import tensorflow as tf
import numpy as np

class LSTM(tf.keras.layers.Layer):
    def __init__(self, initial_hc, output_size):
        super(LSTM, self).__init__()
        self.stateful = True
        self.h = initial_hc
        self.hidden_size = self.h.shape[-1].value
        self.output_size = output_size

    def build(self, input_shape):
        h_size, x_size = self.hidden_size, input_shape[-1].value
        self.wh = self.add_variable('wh', shape=[x_size + h_size, 4 * h_size])
        self.bh = self.add_variable('bh', shape=[4 * h_size])
        self.wy = self.add_variable('wy', shape=[h_size, self.output_size])
        self.by = self.add_variable('by', shape=[self.output_size])

    def call(self, h_below):
        h, c = tf.split(self.h, 2)
        h = tf.reshape(h, shape=(tf.shape(h)[1], tf.shape(h)[2]))
        c = tf.reshape(c, shape=(tf.shape(c)[1], tf.shape(c)[2]))
        v = tf.matmul(tf.concat((h_below, h), 1), self.wh) + self.bh

        #gates
        pi, pf, po, pg = tf.split(v, 4, axis=1)
        i, f, o, g = tf.sigmoid(pi), tf.sigmoid(pf), tf.sigmoid(po), tf.tanh(pg)

        #c and h
        c = f * c + i * g
        h = tf.tanh(c) * o

        #output
        output = tf.matmul(h, self.wy) + self.by
        self.h = tf.concat(([h], [c]), 0)
        return output

# uncomment the following for a test example of the LSTM network
'''
x = tf.placeholder(tf.float32, shape=(4, 2))
y = tf.placeholder(tf.float32, shape=(4, 5))
rnn = LSTM(tf.zeros([2, 4, 3]), 5)
h = rnn.h
out = rnn(x)
h2 = rnn.h
out = rnn(x)
h3 = rnn.h

optimizer = tf.train.AdamOptimizer()
loss = tf.losses.mean_squared_error(out, y)
minimize = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xs = np.random.randn(4, 2)
    ys = np.random.randn(4, 5) / 10

    print(sess.run([h2], feed_dict={x: xs, y: ys})[0])
    print(sess.run([h3], feed_dict={x: xs, y: ys})[0])
    while True:
        print(sess.run([loss, minimize], feed_dict={x: xs, y: ys})[0])
'''
