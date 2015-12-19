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
J = T.fscalar("J")

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
# calling twice np.sum is redundant, as numpy completely flat the matrix
# to a single scalar, it doesn't sum-up on a single dimension.
J = 1.0/m  * T.sum(
                    T.sum(
                            ((0-Y) * T.log(A3)) -
                            ((1-Y) * T.log(1-A3))
                            ))
reg = (reg_param/(2*m))*(T.sum(T.power(W1[1:,:], 2)) +
                   T.sum(T.power(W2[1:,:], 2)))

#theano.printing.pp(J)
J = J + reg


cost_function = theano.function(
    inputs=[W,
            input_layer_size, hidden_layer_size, output_layer_size,
            X, Y, learning_rate, reg_param],
    outputs=J,
    on_unused_input='warn')
