import sys

sys.path.append("..")
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import config as cfg
import matplotlib.pyplot as plt
from LeNet.lenet import Lenet
from choose_best_gpu import set_best_gpu


def main():
  SGD_loss = []
  AvSGD_loss = []
  mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)
  batch_size = cfg.BATCH_SIZE
  parameter_path = cfg.PARAMETER_FILE
  lenet = Lenet()
  max_iter = cfg.MAX_ITER

  sess.run(tf.initialize_all_variables())

  for i in range(max_iter):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      params = {lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]}
      train_accuracy = sess.run(lenet.train_accuracy, feed_dict=params)
      sgd_loss = sess.run(lenet.loss, feed_dict=params)
      SGD_loss.append(sgd_loss)
      for v in tf.trainable_variables():
        if v is not None:
          params[v] = sess.run(lenet.avsgd.get_slot(v, "ax").value())
      avsgd_loss = sess.run(lenet.loss, feed_dict=params)
      AvSGD_loss.append(avsgd_loss)
      print("step %d, training accuracy %g" % (i, train_accuracy))

    if i >= 5000:
      if i == 5000:
        print("Switch to AvSGD")
      sess.run(lenet.avsgd_train_op, feed_dict={lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]})
    else:
      sess.run(lenet.sgd_train_op, feed_dict={lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]})
    # sess.run(lenet.avsgd_train_op, feed_dict={lenet.raw_input_image: batch[0], lenet.raw_input_label: batch[1]})
  # save_path = saver.save(sess, parameter_path)
  sess.close()
  plt.plot(SGD_loss, label="SGD")
  plt.plot(AvSGD_loss, label="AvSGD")
  plt.legend(loc='best')
  plt.xlabel("Steps")
  plt.ylabel("Loss")
  plt.savefig("result.png")
  plt.show()


if __name__ == '__main__':
  set_best_gpu(1)
  main()
