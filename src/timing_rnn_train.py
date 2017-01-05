# Author: Alperen AYDIN
# Date: Dec 19, 2016.

# Training of an RNN designed to detect the different kinds of heartbeats
# in waveforms.


import tensorflow as tf
import cnn_functions as cf
from timing_dataset import timing_dataset

dataset = timing_dataset('../data/set_a_timing.csv')


# Parameters of the loop
LOG_STEP = 10
SAVER_STEP = 100

# Inputs for the network
x = tf.placeholder(tf.float64, [1, dataset.max_length, 1])

y_ = tf.placeholder(tf.float64, [dataset.max_length, 2])

# Defining the LSTM cell
num_hidden = 15
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)

val, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float64)

val = tf.reshape(val, [dataset.max_length, 15])

# We want to map the 15 channel output of LSTM to single channel waveform
y = cf.fc_nn(val, [num_hidden, 2])

# Our loss/energy function is the cross-entropy between the label and the
# output
loss = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# We are using the Adam Optimiser because it is effective at managing the
# learning rate and momentum
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

log = open('logs/timing_rnn.txt', 'a')

saver = tf.train.Saver()
checkpoint = 0

with sess.as_default():
    for s in range(1, int(2e6)):
        data, label = dataset.next_train()
        print 'step {}'.format(s)

        # We update the log with the newest performance results
        if (s % LOG_STEP == 0):
            # We calculate the performance results
            # for the training set on the current batch
            train_loss = loss.eval(feed_dict={x: data, y_: label})

            # For the validation set, we do it on the whole thing
            # The final results are means of the results for each batch
            valid_loss = 0
            batch_count = 0.0
            while True:
                vd, vl = dataset.next_valid()
                if vd == -1:
                    break
                batch_count += 1.0
                valid_loss += loss.eval(feed_dict={x: vd, y_: vl})

            valid_loss = valid_loss / batch_count
            # Adding a new line to the log
            logline = 'Epoch {} Batch {} train_loss {} valid_loss {} \n'
            logline = logline.format(
                dataset.completed_epochs, s, train_loss, valid_loss)
            log.write(logline)
            print logline

        if s % SAVER_STEP == 0:
            path = saver.save(sess, 'checkpoints/timing/rnn__',
                              global_step=checkpoint)
            print "Saved checkpoint to %s" % path
            checkpoint += 1

        train_step.run(feed_dict={x: data, y_: label})
