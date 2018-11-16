import tensorflow as tf
import itertools
import sys
sys.path.append('../nns')
from utils import *
from tensorflow.python import debug as tf_debug

learning_rate = 1e-3
beta1 = 0.9
beta2 = 0.999

batch_size = 32
memory_size = 4
input_length = 2
output_length = input_length
input_size = 1
output_size = input_size
#todo: add input_length, output_*, memory_length
#input_size = 1
#output_size = input_size

x = tf.placeholder(tf.float32, shape=(batch_size, 2 * input_length))
y = tf.placeholder(tf.float32, shape=(batch_size, 2 * output_length))
memory = tf.Variable(tf.constant(0.0, shape=(batch_size, memory_size)))

w_im = tf.Variable(tf.random_normal([input_size, memory_size], stddev=0.01))
b_im = tf.Variable(tf.constant(0.0, shape=(memory_size,)))

w_mm = tf.Variable(tf.random_normal([input_size + memory_size, memory_size], stddev=0.01))
b_mm = tf.Variable(tf.constant(0.0, shape=(memory_size,)))

w_mg = tf.Variable(tf.random_normal([input_size + memory_size, 1], stddev=0.01))
b_mg = tf.Variable(tf.constant(0.0, shape=(1,)))

w_mo = tf.Variable(tf.random_normal([input_size + memory_size, output_size], stddev=0.01))
b_mo = tf.Variable(tf.constant(0.0, shape=(output_size,)))

def cond(output):
    return tf.reduce_any(tf.not_equal(tf.shape(output), tf.shape(y)))

def body(output):
    global memory
    x_now = x[:, tf.shape(output)[1]]
    x_now = tf.reshape(x_now, shape=(batch_size, 1))
    memory_attn = tf.nn.softmax(tf.matmul(x_now, w_im) + b_im)
    selected_memory = memory_attn * memory

    new_input = tf.concat((x_now, selected_memory), axis=1)
    new_memory = tf.matmul(new_input, w_mm) + b_mm

    memory_gate = tf.nn.sigmoid(tf.matmul(new_input, w_mg) + b_mg)
    memory = memory_gate * new_memory + (1 - memory_gate) * memory

    new_output = tf.matmul(new_input, w_mo) + b_mo
    return tf.concat((output, new_output), axis=1)

final_output = tf.while_loop(cond, body, [tf.constant(0.0, shape=(batch_size, 0))], shape_invariants=[tf.TensorShape([batch_size, None])])

optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
loss = tf.losses.mean_squared_error(final_output, y)
minimize = optimizer.minimize(loss)

tr_loss, dev_loss, dev_acc = {}, {}, {}

with tf.Session() as sess:
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    sess.run(tf.global_variables_initializer())
    bits = np.random.randn(batch_size, input_length)
    xs = np.concatenate((bits, np.zeros_like(bits)), axis=1)
    ys = np.concatenate((np.zeros_like(bits), bits), axis=1)
    for i in itertools.count():
        tr_loss[i], _, out = sess.run([loss, minimize, final_output], feed_dict={x: xs, y: ys})
        print(out[0])
        print(ys[0])
        print('------')
        #print(tr_loss[i])

#TODO: for every net, add one more layer
