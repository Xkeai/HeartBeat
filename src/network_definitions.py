import tensorflow as tf

from tensorflow.contrib import rnn


def simple_RNN(x, num_hidden, length):
    # Creating the LSTM cell (I am using the default parameter for now)
    lstm_cell = rnn.BasicLSTMCell(num_hidden)
    # Getting the outputs
    outputs, states = tf.nn.dynamic_rnn(
        lstm_cell, x, dtype=tf.float32)

    return(outputs, states)


def singleRNN_Segmentation(num_hidden, length, learning_rate, batch_size):
    # The Input
    x = tf.placeholder(tf.float32, [batch_size, length, 1])
    # The label/target
    y_ = tf.placeholder(tf.float32, [batch_size, length, 2])

    # Defining the LSTM cell and the dynamic RNN
    outputs, states = simple_RNN(x, num_hidden, length)

    # The output is of shape [batch_size, max_length, num_hidden]
    # We need [batch_size, max_length, 2] so we do matrix mutiplication
    # The shape of matrix is [num_hidden, 2]
    weight = tf.random_normal([1, num_hidden, 2], dtype=tf.float32)
    y = tf.matmul(outputs, weight)

    # Calculating the loss
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y, labels=y_)
    loss = tf.reduce_sum(cross_entropy)
    # I am using Adam as it has built-in learning rate reduction.
    # The moments also help speed up the learning.
    # Plus I am lazy.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

    return (x, y, y_, loss, train_op)
