import tensorflow as tf
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print 'data_size=%d characters, vocab_size=%d' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_layer_size = 100
batch_size = 25 # sequence length

X = tf.placeholder(tf.float32, shape=(batch_size, vocab_size))
y = tf.placeholder(tf.float32, shape=(batch_size, vocab_size))

with tf.variable_scope("rnn-prediction"):
    # Declare variables
    Wxh = tf.get_variable("weights-xh", (vocab_size, hidden_layer_size),
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    Whh = tf.get_variable("weights-hh", (hidden_layer_size, hidden_layer_size),
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    Why = tf.get_variable("weights-hy", (hidden_layer_size, vocab_size),
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer())

    bx = tf.get_variable("bias-x", (1, 1),
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(0.))
    bh = tf.get_variable("bias-h", (1, 1),
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(0.))
    by = tf.get_variable("bias-y", (1, 1),
                        dtype=tf.float32,
                        initializer=tf.random_normal_initializer(0.))
    h_state = tf.get_variable("hidden-state", (hidden_layer_size, vocab_size),
                              dtype=tf.float32,
                              initializer=tf.constant_initializer(0.))
    # Define learning graph
    def calc_next_hidden_state(h_prev, single_input_x):
        a1 = tf.matmul([single_input_x], Wxh) + bx
        b1 = tf.matmul(h_prev, Whh) + bh # bh ??
        h_state = tf.nn.sigmoid(a1+b1)
        #h_state = tf.nn.sigmoid(tf.matmul(single_input_x, Wxh) + bx +
        #                        tf.matmul(h_prev, Whh) + bh) # bh ??
        return h_state
    h_states = tf.scan(calc_next_hidden_state,
                       X, # batch of 25 one-hot vectors
                       initializer=tf.zeros((1, hidden_layer_size)))
                       #initializer=tf.constant_initializer(0.))
    # is was 25, 1, 100 reshaped to 25, 100
    y_preds = tf.matmul(tf.reshape(h_states, (batch_size, hidden_layer_size)), Why) + by
    loss = tf.nn.softmax_cross_entropy_with_logits(y_preds, y)

    # Define sampling of 200 characters
    def sample_next_character(prev_state, _):
        #print prev_state
        prev_output_x = prev_state
    #    return prev_state
    #    prev_output_x = tf.slice(prev_state, (0,0), (1,vocab_size))
    #    prev_h_state = tf.slice(prev_state, (0,vocab_size), (1,hidden_layer_size))
        curr_h_state = tf.nn.sigmoid(tf.matmul(prev_output_x, Wxh) + bx ) #+
    #                                 tf.matmul(prev_h_state, Whh) + bh) # bh ??
        curr_y_pred = tf.nn.softmax(tf.matmul(curr_h_state, Why) + by)
    #    curr_y_pred = tf.matmul(curr_h_state, Why) + by
    #    # TODO: sample character from output, here we don't sample from the distrib
    #    # we only return the most likely
        print tf.argmax(curr_y_pred, 1)
        curr_output_x = tf.one_hot(tf.argmax(curr_y_pred, 1), vocab_size)
    #    curr_output_x = tf.reshape(curr_output_x, (1,vocab_size))
    #    return tf.concat(1, [curr_output_x, curr_h_state])
        print curr_output_x
        tf.pack([curr_output_x, curr_h_state])
        return curr_output_x

    #sample_initializer = tf.get_variable("sample-initializer", (vocab_size, 1))
    #sample_initializer = tf.get_variable("sample-initializer", (1, vocab_size))
    inputs14 = np.array([char_to_ix[ch] for ch in "a"])
    sample_initializer = np.zeros((1, vocab_size), dtype=np.float32)
    sample_initializer[np.arange(1), inputs14] = 1


    samples_operation = tf.scan(sample_next_character,
                                # reshape as the input expeceted to be 200, 1 ,1
                                # 200 because we want to sample 200 characters
                                tf.reshape(tf.zeros((10, 1)), (10, 1, 1)), # batch of 200 unused elements
                                #initializer=tf.concat(1, [sample_initializer, tf.zeros((1, hidden_layer_size))])) # you might not want the hidden state to be zero
                                initializer=sample_initializer) # you might not want the hidden state to be zero

    opt = tf.train.AdamOptimizer()
    #gvs = opt.compute_gradients(loss)
    #capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
    #capped_gvs = tf.clip_by_value(gvs, -5., 5.)
    #opt_operation = opt.apply_gradients(capped_gvs)
    opt_operation = opt.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    epoch = 0
    p = 0 # reading position in the input file
    loop = True
    while loop:
        epoch += 1
        # Prepare inputs (we're sweeping from left to right in steps seq_length long)
        if p + batch_size + 1 >= data_size or epoch == 0:
            loop = False
            continue
            #h_state = tf.zeros((hidden_layer_size, vocab_size)) # reset RNN memory
            #p = 0 # go from start of data

        # Two vectors of integers representing the inputs and the corresponding
        # expected outputs
        inputs = np.array([char_to_ix[ch] for ch in data[p:p+batch_size]])
        targets = np.array([char_to_ix[ch] for ch in data[p+1:p+batch_size+1]])
        X_batch = np.zeros((batch_size, vocab_size))
        y_batch = np.zeros((batch_size, vocab_size))
        X_batch[np.arange(batch_size), inputs] = 1
        y_batch[np.arange(batch_size), targets] = 1

        _, loss_val = sess.run([opt_operation, loss],
                               feed_dict={X: X_batch, y: y_batch})
        if epoch % 100 == 0:
            print "sample"
            # sample a number of characters from the rnn
            # Select X_batch[0] as seed
            #sample_initializer = tf.convert_to_tensor(X_batch[0])
            #sample_initializer = X_batch[0]
            #samples, _ = sess.run([samples_operation])
            samples1 = sess.run([samples_operation])
            sentence = ""
            for samples2 in samples1:
                for sample in samples2:
                    #print sample
                    idx = tf.argmax(tf.squeeze(sample), 0)
                    #print ix_to_char[idx.eval()]
                    #print tf.squeeze(sample)
                    #print idx.eval()
                    sentence += ix_to_char[idx.eval()]
            print sentence
