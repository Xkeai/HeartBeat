#!/usr/bin/env python

# The implementation here is greatly inspired by this example:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

import tensorflow as tf

from network_definitions import singleRNN_Segmentation

import logger


from timingDataset import timingDataset

# Creating the dataset
dataset = timingDataset("../data/set_a_timing.csv")

# Creating the logger object
fields = ["train_step", "epoch", "batch", "train_loss", "valid_loss"]
logFname = "../logs/" + logger.getLogName()
log = logger.LogWriter(logFname, fields)

# Defining some general variables to be used in the graph
# and the learning process

# The number of hidden units for the RNN
num_hidden = 16
# The Learning rate for the optimizer
learning_rate = 10**-3
# The length of the data
data_length = int(1e5)
step_size = int(5e4)

# Some variables to control the training
LOG_STEP = 10
SAVER_STEP = 10
training_steps = 10**4
batch_size = 1

# Creating the network:
x, y, y_, loss, train_op = singleRNN_Segmentation(
    num_hidden, data_length, learning_rate, batch_size)

# The initialiser of the variables in the graph
init = tf.global_variables_initializer()
# Saver for saving checkpoints
saver = tf.train.Saver()
checkpoint = 0

with tf.Session() as sess:
    sess.run(init)

    for s in range(training_steps + 1):
        print("step %d" % s)
        train_data, train_label = dataset.next_batch_train(batch_size,
                                                           data_length=data_length,
                                                           step_size=step_size)
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
            while True:
                valid_data, valid_label, ended = dataset.next_batch_valid(
                    1,
                    data_length=data_length,
                    step_size=step_size)
                if ended == -1:
                    break
                valid_loss += sess.run(
                    loss,
                    feed_dict={x: valid_data, y_: valid_label})
                no_valid += 1

            log_entry["valid_loss"] = valid_loss / no_valid
            log.addEntry(log_entry)
            print(log_entry)

        if(s % SAVER_STEP == 0):
            path = saver.save(sess,
                              "../checkpoints/checkpoint",
                              global_step=checkpoint)
            print("Saved checkpoint to %s" % (path))
