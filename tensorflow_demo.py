# coding=utf-8
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from TensorFlowASGD import AsynchronousStochasticGradientDescent
from choose_best_gpu import set_best_gpu


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.zeros([in_size, out_size], dtype=tf.float64), dtype=tf.float64)
    biases = tf.Variable(tf.zeros([1, out_size], dtype=tf.float64), dtype=tf.float64)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outpus = Wx_plus_b
    else:
        outpus = activation_function(Wx_plus_b)
    return outpus, Weights, biases


set_best_gpu(4)
LR = 0.01
BATCH_SIZE = 40
EPOCH = 12

loss_SGD = []
loss_ASGD = []

x_data = np.expand_dims((np.linspace(-1, 1, 1000)), 1)
np.random.shuffle(x_data)
y_data = np.power(x_data, 2) + 0.1 * np.random.normal(np.zeros(x_data.shape))

x_train_data = x_data[:4 * x_data.size / 5]
y_train_data = y_data[:4 * y_data.size / 5]

x_test_data = x_data[4 * x_data.size / 5:]
y_test_data = y_data[4 * y_data.size / 5:]


def get_batch_data(step):
    return x_train_data[step * BATCH_SIZE:(step + 1) * BATCH_SIZE], \
           y_train_data[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]


def SGD():
    plt.scatter(x_data, y_data)
    plt.show()

    x = tf.placeholder(tf.float64, [None, 1])
    y = tf.placeholder(tf.float64, [None, 1])

    hidden, hidden_W, hidden_b = add_layer(x, 1, 20, activation_function=tf.nn.relu)
    prediction, prediction_W, prediction_b = add_layer(hidden, 20, 1, activation_function=None)

    #
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices=[1]))

    optimizer = tf.train.GradientDescentOptimizer(LR)

    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(EPOCH):
            for step in range(x_train_data.size / BATCH_SIZE):
                b_x, b_y = get_batch_data(step)
                sess.run(train_step, feed_dict={x: b_x, y: b_y})
                # loss_SGD.append(sess.run(loss, feed_dict={x: b_x, y: b_y}))
                loss_SGD.append(sess.run(loss, feed_dict={x: x_test_data, y: y_test_data}))


def ASGD():
    # plt.scatter

    x = tf.placeholder(tf.float64, [None, 1])
    y = tf.placeholder(tf.float64, [None, 1])

    hidden, hidden_W, hidden_b = add_layer(x, 1, 20, activation_function=tf.nn.relu)
    prediction, prediction_W, prediction_b = add_layer(hidden, 20, 1, activation_function=None)

    #
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y - prediction), reduction_indices=[1]))

    optimizer = AsynchronousStochasticGradientDescent(LR, t0=20)

    train_step = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        with tf.summary.FileWriter("./ASGD", sess.graph):
            for epoch in range(EPOCH):
                for step in range(x_train_data.size / BATCH_SIZE):
                    b_x, b_y = get_batch_data(step)
                    sess.run(train_step, feed_dict={x: b_x, y: b_y})
                    # loss_ASGD.append(sess.run(loss, feed_dict={x: b_x, y: b_y}))
                    loss_ASGD.append(sess.run(loss, feed_dict={x: x_test_data,
                                                               y: y_test_data,
                                                               hidden_W: optimizer.get_slot(hidden_W, "ax").eval(),
                                                               hidden_b: optimizer.get_slot(hidden_b, "ax").eval(),
                                                               prediction_W: optimizer.get_slot(prediction_W,
                                                                                                "ax").eval(),
                                                               prediction_b: optimizer.get_slot(prediction_b,
                                                                                                "ax").eval()
                                                               }))
                    for v in tf.trainable_variables():
                        if optimizer.get_slot(v, "ax") is not None:
                            print(optimizer.get_slot(v, "ax"))


SGD()
ASGD()

print(loss_ASGD)
print(loss_SGD)

plt.plot(loss_SGD, label="SGD")
plt.plot(loss_ASGD, label="ASGD")
plt.legend(loc='best')
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.show()
