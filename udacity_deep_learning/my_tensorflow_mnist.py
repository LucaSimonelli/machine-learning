import tensorflow as tf
import math

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

image_size = 784
hidden_layer_size = 1000
output_classes = 10

with tf.variable_scope("rnn-prediction"):
    x_ = tf.placeholder(tf.float32, [None, image_size])

    Wxh = tf.get_variable("weights-xh", (image_size, hidden_layer_size),
                        dtype=tf.float32,
                        #initializer=tf.random_normal_initializer())
                        initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(float(784.0))))
                        #initializer=tf.random_normal_initializer(stddev=2.0/math.sqrt(float(784.0))))
                        #initializer=tf.random_normal_initializer(stddev=1.0/math.sqrt(float(100000.0))))
    bx = tf.get_variable("bias-x", (hidden_layer_size),
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.))

    Why = tf.get_variable("weights-hy", (hidden_layer_size, output_classes),
                        dtype=tf.float32,
                        #initializer=tf.random_normal_initializer())
                        initializer=tf.constant_initializer(0.))
                        #initializer=tf.random_normal_initializer(stddev=1.0/ math.sqrt(float(1000.0))))
    by = tf.get_variable("bias-y", (output_classes),
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.))

    h_states = tf.nn.relu(tf.matmul(x_, Wxh) + bx)
    y = tf.nn.softmax(tf.matmul(h_states, Why) + by)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.scalar_summary('accuracy', tf.reduce_sum(accuracy))
merged = tf.merge_all_summaries()

with tf.Session() as sess:
    train_writer = tf.train.SummaryWriter('./logs2', sess.graph)
    tf.initialize_all_variables().run()

    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        summary, acc, _ = sess.run([merged, accuracy, train], feed_dict={x_: batch_xs, y_: batch_ys})
        train_writer.add_summary(summary, i)
        print acc


    print(sess.run(accuracy, feed_dict={x_: mnist.test.images, y_: mnist.test.labels}))
