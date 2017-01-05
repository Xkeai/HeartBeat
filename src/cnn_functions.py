import tensorflow as tf

# Weight variable


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)

# Bias variable


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float64)
    return tf.Variable(initial)
# These two for the basis for most of the other function


# This convolution has the same size for the output as the input
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# This pooling layer halves the size


def max_pool_2x1(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1],
                          strides=[1, 2, 1, 1], padding='SAME')

# A 2d connvolutional layer with bias


def conv2d_bias(x, shape):
    W_conv = weight_variable(shape)
    b_conv = bias_variable([shape[3]])

    return (conv2d(x, W_conv) + b_conv)

# Everything needed for a convolutional in a single function


def cnm2x1Layer(x, shape):
    h_conv = tf.nn.relu(conv2d_bias(x, shape))
    h_pool = max_pool_2x1(h_conv)
    return h_pool

# A fully connected neural network


def fc_nn(x, shape):
    W_fc = weight_variable(shape)
    b_fc = bias_variable([shape[1]])

    return tf.matmul(x, W_fc) + b_fc
