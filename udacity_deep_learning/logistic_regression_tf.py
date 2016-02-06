import numpy as np
import tensorflow as tf
import cPickle as pickle
import os
from six.moves import range
from process_data import (NotMNIST,
                          DATA_DIR, DATA_PICKLE_FILE, DATA_IMAGE_SIZE, DATA_NUM_LABELS)

MNIST = NotMNIST(pickle_file = os.path.join(DATA_DIR, DATA_PICKLE_FILE),
                 max_train_samples=5000,
                 max_valid_samples=500,
                 max_test_samples=500)

MNIST.verify_data_is_balanced()
MNIST.reshape_dataset()
MNIST.one_hot_labels()
# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
#train_subset = 10000

sgd = True
batch_size = 128
graph = tf.Graph()
with graph.as_default():
    # Input data.
    if sgd:
        # Input data. For the training data, we use a placeholder that will be fed
        # at run time with a training minibatch.
        tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, DATA_IMAGE_SIZE * DATA_IMAGE_SIZE))
        tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, DATA_NUM_LABELS))
    else:
        # Load the training, validation and test data into constants that are
        # attached to the graph.
        tf_train_dataset = tf.constant(MNIST.train_dataset)
        tf_train_labels = tf.constant(MNIST.train_labels)
    tf_valid_dataset = tf.constant(MNIST.valid_dataset)
    tf_test_dataset = tf.constant(MNIST.test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random valued following a (truncated)
    # normal distribution. The biases get initialized to zero.
    weights = tf.Variable(
    tf.truncated_normal([DATA_IMAGE_SIZE * DATA_IMAGE_SIZE, DATA_NUM_LABELS]))
    biases = tf.Variable(tf.zeros([DATA_NUM_LABELS]))

    # Training computation.
    # We multiply the inputs with the weight matrix, and add biases. We compute
    # the softmax and cross-entropy (it's one operation in TensorFlow, because
    # it's very common, and it can be optimized). We take the average of this
    # cross-entropy across all training examples: that's our loss.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    #print logits.get_shape()
    #print tf_train_labels.get_shape()
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    # We are going to find the minimum of this loss using gradient descent.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    # These are not part of training, but merely here so that we can report
    # accuracy figures as we train.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


if sgd:
    num_steps = 3001
    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (MNIST.train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = MNIST.train_dataset[offset:(offset + batch_size), :]
        batch_labels = MNIST.train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), MNIST.valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), MNIST.test_labels))
else:
    num_steps = 801
    with tf.Session(graph=graph) as session:
      # This is a one-time operation which ensures the parameters get initialized as
      # we described in the graph: random weights for the matrix, zeros for the
      # biases. 
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
          print('Loss at step %d: %f' % (step, l))
          print('Training accuracy: %.1f%%' % accuracy(
            predictions, MNIST.train_labels))
          # Calling .eval() on valid_prediction is basically like calling run(), but
          # just to get that one numpy array. Note that it recomputes all its graph
          # dependencies.
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), MNIST.valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), MNIST.test_labels))


