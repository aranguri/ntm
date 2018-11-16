import tensorflow as tf
import itertools
import sys
sys.path.append('../nns')
from utils import *

learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.999

input_size = 4
batch_size = 128
memory_size = 16
output_size = 4

x = tf.placeholder(tf.float32, shape=(batch_size, input_size))
y = tf.placeholder(tf.float32, shape=(batch_size, output_size))
memory = tf.Variable(tf.constant(0.0, shape=(batch_size, memory_size)))

w_im = tf.Variable(tf.random_normal([input_size, memory_size], stddev=0.01))
b_im = tf.Variable(tf.constant(0.0, shape=(memory_size,)))
memory_attn = tf.nn.softmax(tf.matmul(x, w_im) + b_im)
selected_memory = memory_attn * memory

w_mm = tf.Variable(tf.random_normal([input_size + memory_size, memory_size], stddev=0.01))
b_mm = tf.Variable(tf.constant(0.0, shape=(memory_size,)))
new_input = tf.concat((x, selected_memory), axis=1)
new_memory = tf.matmul(new_input, w_mm) + b_mm

w_mg = tf.Variable(tf.random_normal([input_size + memory_size, 1], stddev=0.01))
b_mg = tf.Variable(tf.constant(0.0, shape=(1,)))
memory_gate = tf.nn.sigmoid(tf.matmul(new_input, w_mg) + b_mg)
memory = memory_gate * new_memory + (1 - memory_gate) * memory

w_mo = tf.Variable(tf.random_normal([input_size + memory_size, output_size], stddev=0.01))
b_mo = tf.Variable(tf.constant(0.0, shape=(output_size,)))
output = tf.matmul(new_input, w_mo) + b_mo

def cond(output):
    return tf.equal(tf.size(output), tf.size(y))

def body(output):
    return [tf.add(t1, 1), t2]

#final_output = tf.while_loop(cond, body, [[]])

optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
loss = tf.losses.mean_squared_error(output, y)
minimize = optimizer.minimize(loss)

tr_loss, dev_loss, dev_acc = {}, {}, {}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    xs = np.random.randn(batch_size, input_size)
    ys = np.random.randn(batch_size, input_size)
    for i in itertools.count():
        tr_loss[i], _ = sess.run([loss, minimize], feed_dict={x: xs, y: ys}))

#TODO: for every net, add one more layer
