import tensorflow as tf
import numpy as np

a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

W1 = tf.ones((2, 2))
# the name labels the variable in the computation graph
# it's just a way of indexing the variables in the global computation graph
# Python namespaces and TF namespaces are two different things
W2 = tf.Variable(tf.zeros((2, 2)), name="weights")

state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

# How to feed data into the network
# 1) from numpy variables
na = np.zeros((2,2))
ta = tf.convert_to_tensor(na)
# 2) from placeholders
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    # c.eval() is just syntacic sugar for
    # sess.run(c) in the current active session
    #print(sess.run(c))
    #print(c.eval())

    print(sess.run(W1))
    # a variable is a box, it doesn't exists until you initialise it
    sess.run(tf.initialize_all_variables())
    print(sess.run(W2))

    print(sess.run(state))
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

    print(sess.run(ta))
    print(sess.run([output],
                   feed_dict={input1:[7.], input2:[2.]}))

