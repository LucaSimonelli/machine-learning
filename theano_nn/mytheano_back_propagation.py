import numpy as np
import theano.tensor as T
import theano
from theano.tensor.nnet import sigmoid

# Define input variables
W = T.fvector("W")
input_layer_size = T.lscalar("input_layer_size")
hidden_layer_size = T.lscalar("hidden_layer_size")
output_layer_size = T.lscalar("output_layer_size")
X = T.fmatrix("X")
Y = T.fmatrix("Y")
learning_rate = T.fscalar("learning_rate")
reg_param = T.fscalar("reg_param")
# Define output variables
W_grad = T.fmatrix("W_grad")

# Implement cost function process
splits = T.lvector("splits")
splits = [hidden_layer_size * (input_layer_size+1),
          output_layer_size * (hidden_layer_size + 1)]
(W1, W2) = T.split(W, splits, n_splits=2)
W1 = W1.reshape((hidden_layer_size, input_layer_size+1))
W2 = W2.reshape((output_layer_size, hidden_layer_size+1))
A2 = sigmoid(T.dot(X, W1.transpose()))
# Add to the left of A2 a column of ones
A2 = T.concatenate([T.ones((T.shape(A2)[0], 1), dtype=A2.dtype), A2], axis=1)
A3 = sigmoid(T.dot(A2, W2.transpose()))


# m int - Number of samples for training
m = T.shape(Y)[0]
# m is an int64 that will force 1.0/m to be a float64, that will make W_grad a
# float64 matrix, compromising the theano
# optimization, as the current gpu hardware is good for float32. Cast m from
# int64 to float32 and exploit fully the advantages of the gpu.
m = m.astype(theano.config.floatX)

D3 = A3 - Y
# rows=number of samples, columns=number of hidden units
D2 = T.zeros((T.shape(X)[0], T.shape(A2)[1]), dtype=theano.config.floatX)
D2 = (T.dot(D3, W2) * A2) * (1-A2)
# remove bias column from D2
D2 = D2[:,1:]

W1_grad = T.dot(D2.transpose(), X)
W2_grad = T.dot(D3.transpose(), A2)
T.set_subtensor(W1_grad[:,1:], W1_grad[:,1:] + learning_rate * W1_grad[:,1:])
W1_grad = (1.0/m) * W1_grad
T.set_subtensor(W2_grad[:,1:], W2_grad[:,1:] + (learning_rate * W2_grad[:,1:]))
W2_grad = (1.0/m) * W2_grad
W_grad =  T.concatenate((W1_grad.flatten(), W2_grad.flatten()), axis=0)
back_propagation = theano.function(
    inputs=[W,
            input_layer_size, hidden_layer_size, output_layer_size,
            X, Y, learning_rate, reg_param],
    outputs=W_grad,
    on_unused_input='warn')
