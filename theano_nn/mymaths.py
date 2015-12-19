import numpy as np
from add_ones import add_ones
from scipy.special import expit as sigmoid

def cost_function(W, input_layer_size, hidden_layer_size, output_layer_size,
                  X, Y, learning_rate, reg_param):
    """ cost function
    @param[in] Y numpy matrix of dscalars - One hot encoded matrix with target results
    @param[in] H numpy matrix of dscalars - matrix of outputs of the nn after feedforward pass, each row is an output vector
    @param[in] lmb scalar - lambda value for regularization to avoid overfitting
    @return J dscalar representing the cost
    """
    # unpack original matrices
    (W1, W2, _) = np.split(W, [hidden_layer_size * (input_layer_size+1),
                               hidden_layer_size * (input_layer_size + 1) +
                                output_layer_size * (hidden_layer_size + 1)])
    W1 = W1.reshape((hidden_layer_size, input_layer_size+1))
    W2 = W2.reshape((output_layer_size, hidden_layer_size+1))
    A2 = add_ones(sigmoid(np.dot(X, W1.transpose())))
    A3 = sigmoid(np.dot(A2, W2.transpose()))


    # m int - Number of samples for training
    m = Y.shape[0]
    # calling twice np.sum is redundant, as numpy completely flat the matrix
    # to a single scalar, it doesn't sum-up on a single dimension.
    J = 1.0/m  * np.sum(
                        np.sum(
                                np.multiply(-Y, np.log(A3)) -
                                np.multiply(1-Y, np.log(1-A3))
                                ))
    reg = (reg_param/(2*m))*(np.sum(np.power(W1[1:,:], 2)) +
                       np.sum(np.power(W2[1:,:], 2)))
    print "cost=%.10f" % (J + reg,)
    return J + reg


def back_propagation(W, input_layer_size, hidden_layer_size, output_layer_size,
                     X, Y, learning_rate, reg_param):
    """
    @return gradient
    """
    # unpack original matrices
    (W1, W2, _) = np.split(W, [hidden_layer_size * (input_layer_size+1),
                               hidden_layer_size * (input_layer_size + 1) +
                                output_layer_size * (hidden_layer_size + 1)])
    W1 = W1.reshape((hidden_layer_size, input_layer_size+1))
    W2 = W2.reshape((output_layer_size, hidden_layer_size+1))
    A2 = add_ones(sigmoid(np.dot(X, W1.transpose())))
    A3 = sigmoid(np.dot(A2, W2.transpose()))

    m = Y.shape[0]
    D3 = A3 - Y
    # rows=number of samples, columns=number of hidden units
    D2 = np.zeros((X.shape[0], A2.shape[1]))
    for i in xrange(0, m):
        D2[i] = np.multiply(np.multiply(np.dot(D3[i], W2), A2[i]), 1-A2[i])
    #for i in xrange(0, m):
    #D2 = np.multiply(np.multiply(np.dot(D3, W2), A2), 1-A2)

    # remove bias column from D2
    D2 = D2[:,1:]

    W1_grad = np.dot(D2.transpose(), X)
    W2_grad = np.dot(D3.transpose(), A2)
    W1_grad[:,1:] = W1_grad[:,1:] + np.multiply(learning_rate, W1_grad[:,1:])
    W1_grad = np.multiply(1.0/m, W1_grad)
    W2_grad[:,1:] = W2_grad[:,1:] + np.multiply(learning_rate, W2_grad[:,1:])
    W2_grad = np.multiply(1.0/m, W2_grad)
    return np.concatenate((W1_grad.flatten(), W2_grad.flatten()), axis=0)

def numerical_grad(W1, W2, X, Y):
    """ Check that the gradient computed by back propagation is correct
    """
    e = 0.0001
    # feed forward
    A2 = add_ones(sigmoid(np.dot(X, W1.transpose())))
    A3 = sigmoid(np.dot(A2, W2.transpose()))
    # append the rows from W2 to W1 and make a single matrix
    W = np.concatenate((W1.flatten(), W2.flatten()), axis=0)
    E = np.zeros(W.shape)
    Ng = np.zeros(W.shape) # Numerical Gradient
    for i in xrange(0, E.shape[0]):
        print i
        E[i] = e
        Dm = W - E
        Dp = W + E
        #print "W.shape=", W.shape
        #print "dm.shape=", Dm.shape
        # reshape Dm and Dp
        (DmW1, DmW2, _d1) = np.split(Dm, [W1.size, W1.size+W2.size])
        (DpW1, DpW2, _d2) = np.split(Dp, [W1.size, W1.size+W2.size])
        #print "DmW1.shape=", DmW1.shape
        #print "DmW2.shape=", DmW2.shape
        #print "_d1.shape=", _d1.shape
        DpW1 = DpW1.reshape(W1.shape)
        DpW2 = DpW2.reshape(W2.shape)
        DmW1 = DmW1.reshape(W1.shape)
        DmW2 = DmW2.reshape(W2.shape)
        cost_m = cost_function(DmW1, DmW2, Y, A3)
        cost_p = cost_function(DpW1, DpW2, Y, A3)
        Ng[i] = (cost_p - cost_m) / (2.0 * e)
        E[i] = 0.0
    (NgW1, NgW2, _) = np.split(Ng, (W1.size, W1.size+W2.size))

    NgW1 = NgW1.reshape(W1.shape)
    NgW2 = NgW2.reshape(W2.shape)
    return (NgW1, NgW2)

