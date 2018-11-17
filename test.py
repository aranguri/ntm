import tensorflow as tf
import numpy as np
from rnn_layer import RNN

rnn = RNN(tf.zeros([6, 3]), 3)

def cond(i, rnn_h):
    rnn.h = rnn_h
    return tf.reduce_all([tf.equal(rnn.h[0][0], 0), tf.less(i, 10)])

def body(i, rnn_h):
    rnn.h = rnn_h
    x = tf.random_normal((6, 5))
    rnn(x)
    i = tf.add(i, 1.0)
    return i, rnn.h

final_rnn = tf.while_loop(cond, body, [tf.Variable(tf.constant(0.0)), rnn.h])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([final_rnn]))
