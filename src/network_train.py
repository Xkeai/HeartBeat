#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import scipy.signal as signal
import logger

import os
from classificationDataset import classificationDataset

# Creating the dataset
dataset = classificationDataset("../data/set_b.csv", "../data/", seed=123)

# Creating the folder for training session
sessionPath = "../sessions/test_" + logger.getSessionPath()
os.mkdir(sessionPath)
logPath = sessionPath
checkpointPath = sessionPath + "checkpoints/"
os.mkdir(checkpointPath)

# Creating the logger object

fields = ["train_step", "epoch", "batch", "train_loss",
          "valid_loss", "train_accuracy", "valid_accuracy"]
logFname = logPath + logger.getLogName()
log = logger.LogWriter(logFname, fields)

# Defining some general variables to be used in the graph
# and the learning process

# The number of conv+max_pool layers we will have
n_layers = 4
# The number of filter for each convolutional layer
n_filters = [4, 4, 4, 2]
kernel_size = [16, 8, 4, 2]
kernel_size = [[i, i] for i in kernel_size]
# The Learning rate for the optimizer
learning_rate = 0.01

# Some variables to control the training
LOG_STEP = 10
SAVER_STEP = 10
training_steps = 5 * 10**5
batch_size = 1

# Parameters of preprocessing
nperseg = 2**10
noverlap = 2**10 - 2**7


def preprocess(data):
    # Transforming the sound signal into an image
    t, f, SXX = signal.stft(data,
                            axis=1,
                            nperseg=nperseg,
                            noverlap=noverlap)
    # Feature scaling the image
    # Method used: Standardization
    SXX = np.abs(SXX)
    mu = np.mean(SXX, axis=(1, 2))
    sigma = np.std(SXX, axis=(1, 2))
    if np.any(sigma == 0):
        print "sigma is zero"
    if np.any(np.isnan(sigma)):
        print "sigma is nan"
    for n in range(data.shape[0]):
        SXX[n, :, :] = (SXX[n, :, :] - mu[n]) / sigma[n]
    SXX = SXX[:, :, :, np.newaxis]
    return SXX


# Creating the network:

# The Input
x = tf.placeholder(tf.float32,
                   [None, 513, 872, 1],
                   name="x")
# The label/target
y_ = tf.placeholder(tf.float32,
                    [None, 3],
                    name="y_")

conv_out = x

# The convolution + pooling layers
for i in range(n_layers):
    h_conv = tf.layers.conv2d(
        inputs=conv_out,
        filters=n_filters[i],
        kernel_size=kernel_size[i],
        padding="SAME",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(),
        name=("conv_%d" % i))

    h_pool = tf.layers.max_pooling2d(
        inputs=h_conv,
        pool_size=kernel_size[i],
        strides=(2, 2),
        name=("pool_%d" % i),
        padding="SAME")
    conv_out = h_pool
# Passing the results through a dense layer
h_flat = tf.contrib.layers.flatten(conv_out)
h_dense = tf.layers.dense(h_flat, units=3, activation=tf.nn.relu)
# The prediction
y = tf.nn.softmax(h_dense)
# Calculating the loss
loss = tf.losses.mean_squared_error(labels=y_, predictions=y)
# We are also interested in the accuracy
correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))

# I am using Adam as it has built-in learning rate reduction.
# The moments also help speed up the learning.
# Plus I am lazy.
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)


# The initialiser of the variables in the graph
init = tf.global_variables_initializer()
# Saver for saving checkpoints
saver = tf.train.Saver()
checkpoint = 0


with tf.Session() as sess:
    sess.run(init)

    for s in range(training_steps + 1):
        print("step %d" % s)
        # Getting the next batch
        train_data, train_label = dataset.next_batch_train(
            batch_size)
        # Preprocessing
        train_data = preprocess(train_data)
        # A training step
        sess.run(train_op, feed_dict={x: train_data, y_: train_label})
        if(s % LOG_STEP == 0):
            log_entry = {}
            log_entry["train_step"] = s
            log_entry["epoch"] = dataset.epochs
            log_entry["batch"] = dataset.index_train
            log_entry["train_loss"] = sess.run(
                loss,
                feed_dict={x: train_data,
                           y_: train_label})
            log_entry["train_accuracy"] = sess.run(
                accuracy,
                feed_dict={x: train_data,
                           y_: train_label})
            # A list to record the loss for the validation set
            valid_losses = []
            valid_accuracies = []
            while True:
                valid_data, valid_label = dataset.next_batch_valid(
                    batch_size)
                if valid_data is None:
                    break
                # Preprocessing
                valid_data = preprocess(valid_data)
                # We first calculate the loss then the accuracy
                valid_loss = sess.run(
                    loss,
                    feed_dict={x: valid_data, y_: valid_label})
                valid_losses.append(valid_loss)
                valid_accuracy = sess.run(
                    accuracy,
                    feed_dict={x: valid_data, y_: valid_label})
                valid_accuracies.append(valid_accuracy)

            log_entry["valid_loss"] = sum(valid_losses) / len(valid_losses)
            log_entry["valid_accuracy"] = sum(
                valid_accuracies) / len(valid_accuracies)
            log.addEntry(log_entry)
            print(log_entry)

        if(s % SAVER_STEP == 0):
            path = saver.save(sess,
                              checkpointPath + "checkpoint",
                              global_step=checkpoint)
            print("Saved checkpoint to %s" % (path))
            checkpoint += 1
