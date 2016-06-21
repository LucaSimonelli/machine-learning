import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X_data = np.arange(100, step=.1)
y_data = X_data + 20 * np.sin(X_data/10)

#plt.scatter(X_data, y_data)
#plt.show()

n_samples = 1000
batch_size = 100
X_data = np.reshape(X_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope("nn-regression"):
    W = tf.get_variable("weights", (1, 100),
                        initializer=tf.random_normal_initializer())
    W2 = tf.get_variable("weights2", (100, 1),
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1, 1),
                        initializer=tf.random_normal_initializer(0.))
    b2 = tf.get_variable("bias2", (1, 1),
                        initializer=tf.random_normal_initializer(0.))
    h_layer = tf.nn.sigmoid(tf.matmul(X, W) + b)
    y_pred = tf.matmul(h_layer, W2) + b2
    # Luca: n_samples is 1000 but when loss is invoked we only provide a batch
    # of 100 points.
    loss = tf.reduce_sum((y - y_pred)**2/n_samples)

opt = tf.train.AdamOptimizer()
opt_operation = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    #for epoch in range(1000):
    epoch = 0
    while True:
        epoch += 1
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]
        _, loss_val = sess.run([opt_operation, loss],
                               feed_dict={X: X_batch, y: y_batch})
        #print "W=%f, W2=%f, b=%f" % (W.eval(), W2.eval(), b.eval())
        if epoch % 1000 == 0:
            y_pred2 = sess.run([y_pred], feed_dict={X: X_batch})
            plt.scatter(X_data, y_data, c="blue")
            plt.scatter(X_batch, y_pred2, c="red")
            plt.show()


