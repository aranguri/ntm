import tensorflow as tf
import numpy as np

class RNN(tf.keras.layers.Layer):
    def __init__(self, initial_h, output_size):
        super(RNN, self).__init__()
        self.stateful = True
        self.h = initial_h
        self.hidden_size = initial_h.shape[-1].value
        self.output_size = output_size

    def build(self, input_shape):
        self.w_xh = self.add_variable('w_xh', shape=[input_shape[-1].value, self.hidden_size])
        self.w_hh = self.add_variable('w_hh', shape=[self.hidden_size, self.hidden_size])
        self.b_h = self.add_variable('b_h', shape=[self.hidden_size]) #todo: biases aren't starting in 0.
        self.w_hy = self.add_variable('w_hy', shape=[self.hidden_size, self.output_size])
        self.b_y = self.add_variable('b_y', shape=[self.output_size])

    def call(self, input):
        self.h = tf.nn.tanh(tf.matmul(input, self.w_xh) + tf.matmul(self.h, self.w_hh) + self.b_h)
        return tf.nn.tanh(tf.matmul(self.h, self.w_hy) + self.b_y)

#'''
#Test example:

x = tf.placeholder(tf.float32, shape=(4, 2))
y = tf.placeholder(tf.float32, shape=(4, 5))
rnn = RNN(tf.zeros([4, 3]), 5)
out = rnn(x)

optimizer = tf.train.AdamOptimizer()
loss = tf.losses.mean_squared_error(out, y)
minimize = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xs = np.random.randn(4, 2)
    ys = np.random.randn(4, 5)

    while True:
        print(sess.run([loss, minimize], feed_dict={x: xs, y: ys})[0])
#'''
