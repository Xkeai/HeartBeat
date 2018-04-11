#!/usr/bin/env python

# The implementation here is greatly inspired by this example:
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

import tensorflow as tf

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

# The number of conv+max_pool layers we will have
n_layers = 7
# The number of filter for each convolutional layer
n_filters = [8, 8, 8, 8, 8, 4, 2]
kernel_size = [2**(n_layers - n) for n in range(n_layers)]
# The Learning rate for the optimizer
learning_rate = 0.01
# The length of the data
data_length = 2**13
step_size = 2**11

# Some variables to control the training
LOG_STEP = 10
SAVER_STEP = 10
training_steps = 10**4
batch_size = 1

# Creating the network:

# The Input
x = tf.placeholder(tf.float32,
                   [batch_size, data_length, 1],
                   name="x")
# The label/target
y_ = tf.placeholder(tf.float32,
                    [batch_size, data_length / (2**n_layers), 2],
                    name="y_")

# Reshaping the arguments
xr = tf.reshape(x, shape=[batch_size, data_length, 1, 1])
yr_ = tf.reshape(y_, shape=[batch_size, data_length / (2**n_layers), 1, 2])

conv_out = xr

# The convolution + pooling layers
for i in range(n_layers):
    h_conv = tf.layers.conv2d(
        inputs=conv_out,
        filters=n_filters[i],
        kernel_size=[kernel_size[i], 1],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(),
        name=("conv_%d" % i))

    h_pool = tf.layers.max_pooling2d(
        inputs=h_conv,
        pool_size=[2, 1],
        strides=2,
        name=("pool_%d" % i))
    conv_out = h_pool

# Training operations
y = tf.sigmoid(conv_out, name="y")

# Calculating the loss
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
    logits=conv_out,
    labels=yr_)
loss = tf.reduce_sum(cross_entropy)
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
        train_data, train_label = dataset.next_batch_train(
            batch_size,
            data_length=data_length,
            step_size=step_size,
            label_reduction=2**n_layers)

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
                    batch_size,
                    data_length=data_length,
                    step_size=step_size,
                    label_reduction=2**n_layers)

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
