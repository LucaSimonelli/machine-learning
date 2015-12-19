import sys
import logging
import struct
import time
import numpy as np
import theano

logging.basicConfig()
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)
logger.setLevel(logging.WARNING)


def load_labels_from_file(file_name):
    """
    load labels from file into a vector
    """
    logger.debug("reading file %s" % file_name)
    Y = []
    with open(file_name, "rb") as fd:
        magic_number = struct.unpack('>i', fd.read(4))[0]
        logger.debug("magic number=%d" % magic_number)
        number_of_items = struct.unpack('>i', fd.read(4))[0]
        logger.debug("number of labels=%d" % number_of_items)
        while number_of_items > 0:
            label = struct.unpack('B', fd.read(1))[0]
            #logger.debug("label=%d" % label)
            Y.append(label)
            number_of_items -= 1
    return np.array(Y, dtype=theano.config.floatX)


def show_image(raw_image):
    import matplotlib.pyplot as plt
    plt.imshow(np.reshape(raw_image, newshape=(28,28)))
    plt.gray()
    plt.show()


def load_images_from_file(file_name):
    """
    load input from file into a vector
    """
    logger.debug("reading file %s" % file_name)
    X = []
    with open(file_name, "rb") as fd:
        magic_number = struct.unpack('>i', fd.read(4))[0]
        logger.debug("magic number=%d" % magic_number)
        number_of_items = struct.unpack('>i', fd.read(4))[0] # images
        logger.debug("number of items=%d" % number_of_items)
        number_of_rows = struct.unpack('>i', fd.read(4))[0]
        number_of_columns = struct.unpack('>i', fd.read(4))[0]
        pixels_to_read = number_of_rows * number_of_columns
        while number_of_items > 0:
            image = struct.unpack(str(pixels_to_read)+'B', fd.read(pixels_to_read))
            #show_image(image)
            X.append(image)
            number_of_items -= 1
    return np.array(X, dtype=theano.config.floatX)


def main():

    if len(sys.argv) < 3:
        logger.error("Expected 2 file names in input")
        return 1
    Y = load_labels_from_file(file_name=sys.argv[1])
    X = load_images_from_file(file_name=sys.argv[2])
    import random
    logger.info("Display random image...")
    show_image(random.choice(X))
    #logger.debug("Y=%s", Y)
    logger.debug("X=%s", X)
    return 0


if __name__ == "__main__":
    sys.exit(main())
