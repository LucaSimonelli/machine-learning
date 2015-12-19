import numpy as np
import theano

def random_weights(columns, rows):
    # where 0.12 is just an initialization constant.
    # the first argument of rand is the row number, the second argument is the
    # columns number.
    ret = np.random.rand(rows, columns) * 2 * 0.12 - 0.12
    return ret.astype(theano.config.floatX)

def debug_init_weights(columns, rows):
    return np.reshape(np.sin(np.arange(start=1,
                                       stop=rows*columns+1,
                                       step=1)),
                      (rows, columns)) / 10.0
