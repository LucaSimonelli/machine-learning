from load_input import load_labels_from_file, load_images_from_file
from add_ones import add_ones, one_hot
import NN

import sys
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#logger.setLevel(logging.WARNING)


def main():
    if len(sys.argv) < 5:
        logger.error("Expected 4 file names in input")
        return 1

    Y = load_labels_from_file(file_name=sys.argv[1])
    X = load_images_from_file(file_name=sys.argv[2])
    Y1 = one_hot(Y)
    X1 = add_ones(X)

    Y_test = load_labels_from_file(file_name=sys.argv[3])
    X_test = load_images_from_file(file_name=sys.argv[4])
    Y1_test = one_hot(Y_test)
    X1_test = add_ones(X_test)

    input_layer_size = 28 * 28
    hidden_layer_size = 300
    output_layer_size = 10 # number of labels in output
    nn = NN.NN(input_layer_size, hidden_layer_size, output_layer_size)
    nn.test(X1_test,Y_test)
    nn.train2(X1, Y1)
    nn.test(X1_test,Y_test)
    #import random
    #logger.info("Display random image...")
    #show_image(random.choice(X))
    #logger.debug("Y=%s", Y)
    #logger.debug("X=%s", X)
    return 0

if __name__ == "__main__":
    sys.exit(main())
