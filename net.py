import tensorflow as tf
import itertools
from utils import *
from lstm_layer import LSTM

learning_rate = 1e-1
beta1 = 0.9
beta2 = 0.999
memory_enabled = True

batch_size = 32
memory_size = 128
input_length = 128
output_length = 128
input_size = 1
output_size = input_size
h_size_im = h_size_mm = h_size_mg = h_size_mo = 64

x = tf.placeholder(tf.float32, shape=(batch_size, input_length + output_length))
y = tf.placeholder(tf.float32, shape=(batch_size, input_length + output_length))
initial_memories = tf.Variable(tf.constant(0.0, shape=(1, batch_size, memory_size)))
initial_memories = tf.stop_gradient(initial_memories) #Avoid backprop on initial memory

rnn_im = LSTM(tf.zeros([2, batch_size, h_size_im]), memory_size)
rnn_mm = LSTM(tf.zeros([2, batch_size, h_size_mm]), memory_size)
rnn_mg = LSTM(tf.zeros([2, batch_size, h_size_mg]), memory_size)
rnn_mo = LSTM(tf.zeros([2, batch_size, h_size_mo]), output_size)

def cond(output, memories, *rnn_hs):
    return tf.reduce_any(tf.not_equal(tf.shape(output), tf.shape(y)))

def body(output, memories, *rnn_hs):
    rnn_im.h, rnn_mm.h, rnn_mg.h, rnn_mo.h = rnn_hs
    # the memories variable stores the history of all previous states of the memory
    x_now = x[:, tf.shape(output)[1]]
    x_now = tf.reshape(x_now, shape=(batch_size, 1))

    if memory_enabled:
        memory = memories[-1]

        input_and_all_memory = tf.concat((x_now, memory), axis=1)
        memory_attn = tf.nn.softmax(rnn_im(input_and_all_memory))
        selected_memory = memory_attn * memory

        input_memory = tf.concat((x_now, selected_memory), axis=1)
        new_memory = rnn_mm(input_memory)

        memory_gate = tf.nn.softmax(rnn_mg(input_memory))
        memory = memory_gate * new_memory + (1 - memory_gate) * memory

        memory = tf.reshape(memory, shape=(1, batch_size, memory_size))
        memories = tf.concat((memory, memory), axis=0)

        extra_output = rnn_mo(input_memory)
    else:
        hidden_output = rnn_im(x_now)
        extra_output = rnn_mo(hidden_output)

    new_output = tf.concat((output, extra_output), axis=1)
    return new_output, memories, rnn_im.h, rnn_mm.h, rnn_mg.h, rnn_mo.h

#is there a better way of doing this?
shapes = [tf.TensorShape([batch_size, None]), tf.TensorShape([None, batch_size, memory_size]), tf.TensorShape([2, batch_size, h_size_im]),
          tf.TensorShape([2, batch_size, h_size_mm]), tf.TensorShape([2, batch_size, h_size_mg]), tf.TensorShape([2, batch_size, h_size_mo])]
last_state = tf.while_loop(cond, body, [tf.constant(0.0, shape=(batch_size, 0)), initial_memories, rnn_im.h, rnn_mm.h, rnn_mg.h, rnn_mo.h], shape_invariants=[*shapes])
output, memories = last_state[0:2]

optimizer = tf.train.AdamOptimizer(learning_rate, beta1, beta2)
loss = tf.losses.mean_squared_error(output, y)
minimize = optimizer.minimize(loss)

tf.summary.scalar('loss', loss)
tr_loss = {}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('./logs/9/train ', sess.graph)

    for i in itertools.count():
        bits = np.random.randn(batch_size, input_length)
        xs = np.concatenate((bits, np.zeros_like(bits)), axis=1)
        ys = np.concatenate((np.zeros_like(bits), bits), axis=1)
        tr_loss[i], _, output_ = sess.run([loss, minimize, output], feed_dict={x: xs, y: ys})
        print('Loss', tr_loss[i])
        print(output_[0], ys[0])

        if i % 25 == 0:
            merge = tf.summary.merge_all()
            memories_, summary, output_ = sess.run([memories, merge, output], feed_dict={x: xs, y: ys})
            train_writer.add_summary(summary, i)
            plot(tr_loss)
            '''
            # Debugging
            print('Prediction', out[0])
            print('Real', ys[0])
            print('')
            plt.ion()
            plt.cla()
            plt.imshow(memories_[:-1, 0, :])
            plt.pause(1e-8)
            '''

'''
# Next steps
* enable memories larger than only one real number (ie, increment input_size, output_size, memory_size)
* other task: send n real numbers, then send noise, and then ask for the n inputs.
* is it good to let the neural net use distributed representation for memories? or is it better to apply a sharp softargmax that forces the nn to modify only so many memory slots at a time.
* how does the performance difference between rnn w memory and rnn wo memory change as we vary the input_length?
* Do we want to backpropagate through the memory? Is it better to have a fixed initial memory or the optimal one selected by the nn?
* rnn_im could also receive the memory as input
* add multiple read/write in one iteration if nn wishes to do so
* why aren't relu units used for lstm nets?

# Story

# Others
print_op = tf.Print([memory, memory_gate], [memory[0], memory_gate[0]])
with tf.control_dependencies([print_op]):
    new_output = tf.identity(new_output)
'''
