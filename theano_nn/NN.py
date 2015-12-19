from random_weights import random_weights, debug_init_weights
#from mymaths import back_propagation, numerical_grad
from mytheano_back_propagation import back_propagation
#from mymaths import cost_function
from mytheano_cost_function import cost_function
from add_ones import add_ones
import numpy as np
from scipy.special import expit as sigmoid
from scipy.optimize import fmin_cg
import sys
import theano


class NN(object):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        self.input_layer_size = input_layer_size
        self.hidden_layer_size = hidden_layer_size
        self.output_layer_size = output_layer_size
        # init weights between input layer and hidden layer
        # add 1 to input layer size because of bias theta0
        self.W1 = random_weights(columns=input_layer_size + 1,
                                 rows=hidden_layer_size)
        # init weights between hidden layer and output layer
        self.W2 = random_weights(columns=hidden_layer_size + 1,
                                 rows=output_layer_size)
        self.learning_rate = 3.0
        self.reg_param = 3.0

    def test(self, X, Y):
        A2 = add_ones(sigmoid(np.dot(X, self.W1.transpose())))
        A3 = sigmoid(np.dot(A2, self.W2.transpose()))
        failed = np.count_nonzero(A3.argmax(axis=1)-Y)
        accuracy = (Y.shape[0]-failed)/float(Y.shape[0])
        print "accuracy=%.2f" % np.multiply(accuracy, 100)

    def train(self, X, Y):
        """ assume each row in X is already prefixed with 1, the theta0 bias
        @param[in] X
        @param[in] Y one-hot encoded matrix where each row is a target vector
        """
        for i in xrange(0, 100):
            (cost, W1_grad, W2_grad) = self.cost_and_grad(X, Y)
            self.W1 -= W1_grad
            self.W2 -= W2_grad

    def train2(self, X, Y):
        W = np.concatenate((self.W1.flatten(), self.W2.flatten()), axis=0).astype(theano.config.floatX)
        arguments = (self.input_layer_size,
                     self.hidden_layer_size,
                     self.output_layer_size,
                     X, Y,
                     self.learning_rate, self.reg_param)
        W = fmin_cg(cost_function, W,
                    fprime=back_propagation,
                    args=arguments,
                    maxiter=100)
        (W1, W2, _) = np.split(W, [self.hidden_layer_size * (self.input_layer_size+1),
                               self.hidden_layer_size * (self.input_layer_size + 1) +
                                self.output_layer_size * (self.hidden_layer_size + 1)])
        self.W1 = W1.reshape((self.hidden_layer_size, self.input_layer_size+1))
        self.W2 = W2.reshape((self.output_layer_size, self.hidden_layer_size+1))

    def cost_and_grad(self, X, Y):
        # each row of A2 contains the value of the hidden units for the
        # corresponding input array from X. Also extend A2 to include the bias
        # for the next step.
        #A2 = add_ones(sigmoid(np.dot(X, self.W1.transpose())))
        #A3 = sigmoid(np.dot(A2, self.W2.transpose()))

        # Pack W1 and W2 in 1-d array, provide sizes to the function
        # in a way that it will be able to unpack
        W = np.concatenate((self.W1.flatten(), self.W2.flatten()), axis=0)

        cost = cost_function(W, self.input_layer_size,
                                self.hidden_layer_size,
                                self.output_layer_size,
                                X, Y, #A2, A3,
                                self.learning_rate, self.reg_param)
        print "cost=%.10f" % (cost,)
        W_grad = back_propagation(W, self.input_layer_size,
                                     self.hidden_layer_size,
                                     self.output_layer_size,
                                     X, Y, #A2, A3,
                                     self.learning_rate, self.reg_param)
        ##################################
        # Following gradient check is only for testing
        ##################################
        #(NgW1, NgW2) = numerical_grad(self.W1, self.W2, X, Y)
        #diff_grad_W1 = np.linalg.norm(NgW1-W1_grad, ord=2)/np.linalg.norm(NgW1+W1_grad, ord=2)

        #diff_grad_W2 = np.linalg.norm(NgW2-W2_grad, ord=2)/np.linalg.norm(NgW2+W2_grad, ord=2)
        #print "diff1=%.10f, diff2=%.10f" % (diff_grad_W1, diff_grad_W2)
        ##################################
        # End gradient check
        ##################################
        (W1_grad, W2_grad, _) = np.split(W_grad, [self.hidden_layer_size * (self.input_layer_size+1),
                               self.hidden_layer_size * (self.input_layer_size + 1) +
                                self.output_layer_size * (self.hidden_layer_size + 1)])
        W1_grad = W1_grad.reshape((self.hidden_layer_size, self.input_layer_size+1))
        W2_grad = W2_grad.reshape((self.output_layer_size, self.hidden_layer_size+1))

        return (cost, W1_grad, W2_grad)
