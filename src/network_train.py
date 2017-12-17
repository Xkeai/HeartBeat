#!/usr/bin/env python

# The implementation here is greatly inspired by this example:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

import tensorflow as tf
import logger

from timingDataset import timingDataset
from tensorflow.contrib import rnn

# Creating the dataset
dataset = timingDataset("../data/set_a_timing.csv")

# Creating the logger object
fields = ["train_step", "epoch", "batch", "train_loss", "valid_loss"]
logFname = "../logs/" + logger.getLogName()
log = logger.LogWriter(logFname, fields)

# Defining some general variables to be used in the graph
# and the learning process

num_hidden = 16
learning_rate = 0.001

LOG_STEP = 10
SAVER_STEP = 10
training_steps = 10**4
batch_size = 1

# Creating the network:
# Defining the input

# The data
x = tf.placeholder(tf.float16, [None, dataset.max_length, 1])
# The label/target
y_ = tf.placeholder(tf.float16, [None, dataset.max_length, 2])


# Defining the LSTM cell and the dynamic RNN

# We first have to transform out [batch_size, max_length, 1] shaped input
# into a list of [batch_size, 1] tensor of length [max_length]
x_unstacked = tf.unstack(x, dataset.max_length, 1)
# Creating the LSTM cell (I am using the default parameter for now)
lstm_cell = rnn.BasicLSTMCell(num_hidden)
# Getting the outputs
outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float16)
# The output is of shape [batch_size, max_length, num_hidden]
# We need [batch_size, max_length, 2] so we do matrix mutiplication
# The shape of matrix is [num_hidden, 2]
weight = tf.random_normal([1, num_hidden, 2], dtype=tf.float16)
y = tf.matmul(outputs, weight)

netOutput = tf.nn.softmax(y)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=y, labels=y_))
# I am using Adam as it has built-in learning rate reduction.
# The moments also help speed up the learning.
# Plus I am lazy.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)


# There is a small problem with error.
# I am unsure about how to define the error.
# I am going to think it through and figure out a precise and proper
# definition.

# The initialiser of the variables in the graph
init = tf.global_variables_initializer()
# Saver for saving checkpoints
saver = tf.train.Saver()
checkpoint = 0

with tf.Session() as sess:
    sess.run(init)

    for s in range(training_steps + 1):
        train_data, train_label = dataset.next_batch_train(batch_size)
        sess.run(train_op, feed_dict={x: train_data, y_: train_label})

        if(s % LOG_STEP == 0):
            log_entry = {}
            log_entry["train_step"] = s
            log_entry["epoch"] = dataset.epochs
            log_entry["batch"] = dataset.index_train
            log_entry["train_loss"] = sess.run(
                loss,
                feed_dict={x: train_data, y_: train_label})
            valid_loss = 0
            no_valid = 0
            valid_data, valid_label = dataset.next_batch_valid(1)
            while valid_data != -1:
                valid_loss += sess.run(
                    loss,
                    feed_dict={x: valid_data, y_: valid_label})
                no_valid += 1
                valid_data, valid_label = dataset.next_batch_valid(1)
            log_entry["valid_loss"] = valid_loss / no_valid
            log.addEntry(log_entry)
            print(log_entry)

        if(s % SAVER_STEP == 0):
            path = saver.save(sess,
                              "../checkpoints",
                              global_step=checkpoint)
            print("Saved checkpoint to %s" % (path))
