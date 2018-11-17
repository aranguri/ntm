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

def cond(output, rnn_im_h, rnn_mm_h, rnn_mg_h, rnn_mo_h):
    return tf.reduce_any(tf.not_equal(tf.shape(output), tf.shape(y)))

def body(output, rnn_im_h, rnn_mm_h, rnn_mg_h, rnn_mo_h):
    global memory
    rnn_im.h, rnn_mm.h, rnn_mg.h, rnn_mo.h = rnn_im_h, rnn_mm_h, rnn_mg_h, rnn_mo_h
    x_now = x[:, tf.shape(output)[1]]
    x_now = tf.reshape(x_now, shape=(batch_size, 1))

    memory_attn = tf.nn.softmax(rnn_im(x_now))
    print_op = tf.Print(rnn_im.h, [rnn_im.h])
    selected_memory = memory_attn * memory

    input_memory = tf.concat((x_now, selected_memory), axis=1)
    new_memory = rnn_mm(input_memory)

    memory_gate = tf.nn.softmax(rnn_mg(input_memory))
    memory = memory_gate * new_memory + (1 - memory_gate) * memory

    extra_output = rnn_mo(input_memory)
    new_output = tf.concat((output, extra_output), axis=1)

    #assert_op = tf.Assert(tf.reduce_all(tf.equal(rnn_im.h, tf.zeros_like(rnn_im.h))), [rnn_im.h], name='assert_out_positive')
    #new_output = tf.control_dependencies([assert_op], new_output)
    #with tf.control_dependencies([print_op]):
    #    new_output = tf.identity(new_output)

    return new_output, rnn_im.h, rnn_mm.h, rnn_mg.h, rnn_mo.h

#rnns_h = rnn_im.h, rnn_mm.h, rnn_mg.h, rnn_mo.h
#final_output = tf.while_loop(cond, body, [tf.constant(0.0, shape=(batch_size, 0))], shape_invariants=[tf.TensorShape([batch_size, None])])
shapes = [tf.TensorShape([batch_size, None]), tf.TensorShape([batch_size, memory_size])] + [tf.TensorShape([batch_size, input_size + memory_size])] * 3
last_state = tf.while_loop(cond, body, [tf.constant(0.0, shape=(batch_size, 0)), rnn_im.h, rnn_mm.h, rnn_mg.h, rnn_mo.h], shape_invariants=[*shapes])
final_output = last_state[0]

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
    #ps(sess.run([rnns], feed_dict={x: xs, y: ys}))
        tr_loss[i], _, out = sess.run([loss, minimize, final_output], feed_dict={x: xs, y: ys})
        print(out[0])
        print(ys[0])
        print(tr_loss[i])
    '''
    print(out[0])
    print(ys[0])
    print('------')
    '''

'''
#Next step: the state of the rnns is not working. (we know that because it outputs the same when it receives 0 as input.)
#TODO: tensorboard

Story:
First we start with a feedforward neural net. But then I realize we need a loop over that fnn - otherwise, we aren't taking into account that the task is sequential.
Then, i realize we need a state for the neural net. This happens because the nn receives 0 as input when it's supposed to write. But then, it doesn't know if it's starting to write, or anywhere in the middle
Thus, it has to have an state.
'''
