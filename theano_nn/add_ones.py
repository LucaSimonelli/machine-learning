import numpy as np
import theano

def add_ones(matrix):
    """ extend matrix with a first column of ones """
    matrix_result = np.ones((matrix.shape[0],matrix.shape[1]+1))
    matrix_result[:,1:] = matrix
    return matrix_result.astype(theano.config.floatX)

def one_hot(array, n_values='auto'):
    """ transform the array in input to a one-hot encoded matrix """
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(n_values=n_values, sparse=False)
    array_tmp = np.reshape(array, (array.shape[0], 1))
    enc.fit(array_tmp)
    return enc.transform(array_tmp).astype(theano.config.floatX)
