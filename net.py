import tensorflow as tf
import itertools
import sys
sys.path.append('../nns')
from utils import *
from rnn_layer import RNN

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

rnn_im = RNN(tf.zeros([batch_size, memory_size]), memory_size)
rnn_mm = RNN(tf.zeros([batch_size, input_size + memory_size]), memory_size)
rnn_mg = RNN(tf.zeros([batch_size, input_size + memory_size]), memory_size)
rnn_mo = RNN(tf.zeros([batch_size, input_size + memory_size]), output_size)

def cond(output):
    return tf.reduce_any(tf.not_equal(tf.shape(output), tf.shape(y)))

def body(output):
    global memory
    x_now = x[:, tf.shape(output)[1]]
    x_now = tf.reshape(x_now, shape=(batch_size, 1))
    memory_attn = tf.nn.softmax(rnn_im(x_now))
    selected_memory = memory_attn * memory

    input_memory = tf.concat((x_now, selected_memory), axis=1)
    new_memory = rnn_mm(input_memory)

    memory_gate = tf.nn.softmax(rnn_mg(input_memory))
    memory = memory_gate * new_memory + (1 - memory_gate) * memory

    new_output = rnn_mo(input_memory)
    return tf.concat((output, new_output), axis=1)

final_output = tf.while_loop(cond, body, [tf.constant(0.0, shape=(batch_size, 0))], shape_invariants=[tf.TensorShape([batch_size, None])])

optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
loss = tf.losses.mean_squared_error(final_output, y)
minimize = optimizer.minimize(loss)

tr_loss, dev_loss, dev_acc = {}, {}, {}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    bits = np.random.randn(batch_size, input_length)
    xs = np.concatenate((bits, np.zeros_like(bits)), axis=1)
    ys = np.concatenate((np.zeros_like(bits), bits), axis=1)
    for i in itertools.count():
        print(sess.run([a], feed_dict={x: xs, y: ys}))
        '''
        tr_loss[i], _, out = sess.run([loss, minimize, final_output], feed_dict={x: xs, y: ys})
        print(out[0])
        print(ys[0])
        print('------')
        '''
        #print(tr_loss[i])

'''
#Next step: the state of the rnns is not working. (we know that because it outputs the same when it receives 0 as input.)
#TODO: tensorboard

Story:
First we start with a feedforward neural net. But then I realize we need a loop over that fnn - otherwise, we aren't taking into account that the task is sequential.
Then, i realize we need a state for the neural net. This happens because the nn receives 0 as input when it's supposed to write. But then, it doesn't know if it's starting to write, or anywhere in the middle
Thus, it has to have an state.
'''
