import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from timingDataset import timingDataset


# Getting the dataset
dataset = timingDataset("../data/set_a_timing.csv", normalise=False)

# Some variables

num_hidden = 16
learning_rate = 10**-6
batch_size = 1

diff_n = 0
# Getting one samples
data, label = dataset.next_batch_train(
    batch_size=1,
    data_length=2**13,
    step_size=2**11,
    label_reduction=2**3)

# Transforming the data

t = range(data.shape[1])
s1_loc = 2 * np.where(label[0, :, 0] == 1)[0]
s2_loc = 2 * np.where(label[0, :, 1] == 1)[0]

# Plotting
ax = plt.subplot(2, 1, 1)
ax.plot(t, data[0, :, 0], 'black')
ax.grid()

plt.show()

# We are also going to feed the data into a network and check the result
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('../checkpoints/checkpoint-0.meta')
    saver.restore(sess, tf.train.latest_checkpoint('../checkpoints'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    y = graph.get_tensor_by_name("y:0")

    net_out = sess.run(y, feed_dict={x: data})


print(net_out.shape)
print(label.shape)

# Plotting the output
ax = plt.subplot(2, 2, 1)
t = range(net_out.shape[1])
ax.plot(t, net_out[0, :, 0, 0], color="red")


ax = plt.subplot(2, 2, 2)
ax.plot(t, label[0, :, 0], color="red")


ax = plt.subplot(2, 2, 3)
ax.plot(t, net_out[0, :, 0, 1], color="blue")

ax = plt.subplot(2, 2, 4)
ax.plot(t, label[0, :, 1], color="blue")

plt.show()
