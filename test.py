# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from TensorFlowASGD import AsynchronousStochasticGradientDescent
from choose_best_gpu import set_best_gpu

set_best_gpu(4)


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.zeros([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outpus = Wx_plus_b
    else:
        outpus = activation_function(Wx_plus_b)
    return outpus


LR = 0.01
BATCH_SIZE = 32
EPOCH = 12


def ASGD():
    x_data = np.expand_dims((np.linspace(-1, 1, 1000)), 1)
    y_data = np.power(x_data, 2) + 0.1 * np.random.normal(np.zeros(x_data.shape))

    # plt.scatter

    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    hidden = add_layer(x, 1, 10, activation_function=tf.nn.relu)
    prediction = add_layer(hidden, 10, 1, activation_function=None)

    #
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices=[1]))

    optimizer = AsynchronousStochasticGradientDescent(LR)

    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    l_his = []

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(EPOCH):
            sess.run(train_step, feed_dict={x: x_data, y: y_data})
            print epoch, sess.run(loss, feed_dict={x: x_data, y: y_data})
# plt.plot(l_his, label="SGD")
# plt.legend(loc='best')
# plt.xlabel("Steps")
# plt.ylabel("Loss")
# plt.show()
