import numpy as np
import os
import sys
import tarfile
import urllib
#from IPython.display import display_png, Image
from PIL import Image
from scipy import ndimage
#from sklearn.linear_model import LogisticRegression
import cPickle as pickle
import random


DATA_URL = 'http://yaroslavvb.com/upload/notMNIST/'
DATA_DIR = './data/'
DATA_PICKLE_FILE = 'notMNIST.pickle'
DATA_IMAGE_SIZE = 28
DATA_NUM_LABELS = 10


np.random.seed()


def maybe_download(url, filename, dst_dir):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    if not os.path.exists(dst_dir+filename):
        filename, _ = urllib.urlretrieve(url + filename, dst_dir + filename)
        statinfo = os.stat(filename)
        print "Data downloaded: filename=%s, size=%d" % (filename, statinfo.st_size)
    return dst_dir+filename


def maybe_extract(filename, dst_dir):
    """ extract tar.gz file"""
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if not os.path.exists(root):
        tar = tarfile.open(filename)
        tar.extractall(path=dst_dir)
        tar.close()
    data_folders = [os.path.join(root, d) for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    print data_folders
    return data_folders


def display_random_image(folders):
    """ pic a random image in the data_set_path and display it"""
    folder = random.choice(folders)
    filename = random.choice(os.listdir(folder))
    image_file_path = os.path.join(folder, filename)
    print "Display image: %s" % (image_file_path, )
    image = Image.open(image_file_path)
    image.show()


def load(data_folders, image_size, pixel_depth, min_num_images, max_num_images):
    dataset = np.ndarray(
        shape=(max_num_images, image_size, image_size), dtype=np.float32)
    labels = np.ndarray(shape=(max_num_images), dtype=np.int32)
    label_index = 0
    image_index = 0
    for folder in data_folders:
        print folder
        for image in os.listdir(folder):
            if image_index >= max_num_images:
                raise Exception('More images than expected: %d >= %d' % (
                                image_index, max_num_images))
            image_file = os.path.join(folder, image)
            try:
                image_data = (ndimage.imread(image_file).astype(float) -
                              pixel_depth / 2) / pixel_depth
                if image_data.shape != (image_size, image_size):
                    raise Exception('Unexpected image shape: %s' % str(image_data.shape))
                dataset[image_index, :, :] = image_data
                labels[image_index] = label_index
                image_index += 1
            except IOError as exc:
                print 'Could not read:', image_file, ':', str(exc), '- it\'s ok, skipping.'
        label_index += 1
    num_images = image_index
    dataset = dataset[0:num_images, :, :]
    labels = labels[0:num_images]
    if num_images < min_num_images:
        raise Exception('Many fewer images than expected: %d < %d' % (
                        num_images, min_num_images))
    print 'Full dataset tensor:', dataset.shape
    print 'Mean:', np.mean(dataset)
    print 'Standard deviation:', np.std(dataset)
    print 'Labels:', labels.shape
    return dataset, labels


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def save_data(filename, train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
    try:
      file_obj = open(filename, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
      }
      pickle.dump(save, file_obj, pickle.HIGHEST_PROTOCOL)
      file_obj.close()
    except Exception as exc:
      print 'Unable to save data to', filename, ':', str(exc)
      raise

def load_data(pickle_file,
              max_train_samples, max_test_samples, max_valid_samples):
    file_obj = open(pickle_file, 'rb')
    data = pickle.load(file_obj)
    file_obj.close()
    data['train_dataset'], data['train_labels'] = randomize(data['train_dataset'], data['train_labels'])
    data['valid_dataset'], data['valid_labels'] = randomize(data['valid_dataset'], data['valid_labels'])
    data['test_dataset'], data['test_labels'] = randomize(data['test_dataset'], data['test_labels'])

    train_dataset = data['train_dataset'][:max_train_samples, :, :]
    train_labels = data['train_labels'][:max_train_samples]
    test_dataset = data['test_dataset'][:max_test_samples, :, :]
    test_labels = data['test_labels'][:max_test_samples]
    valid_dataset = data['valid_dataset'][:max_valid_samples, :, :]
    valid_labels = data['valid_labels'][:max_valid_samples]

    return (train_dataset, train_labels,
            valid_dataset, valid_labels,
            test_dataset, test_labels)

def reshape_dataset(dataset):
    """ """
    samples_count, size_x, size_y = dataset.shape
    dataset = dataset.reshape((samples_count, size_x * size_y))
    return dataset

def one_hot_labels(labels):
    """ """
    labels = (np.arange(DATA_NUM_LABELS) == labels[:, None]).astype(np.float32)
    return labels


class NotMNIST(object):
    """
    """
    def __init__(self, pickle_file,
                 max_train_samples,
                 max_valid_samples,
                 max_test_samples):
        """ """
        self.pickle_file = pickle_file
        if not os.path.exists(self.pickle_file):
            self.__generate_pickle_file()

        (self.train_dataset, self.train_labels,
         self.valid_dataset, self.valid_labels,
         self.test_dataset, self.test_labels) = load_data(
                                        pickle_file=self.pickle_file,
                                        max_train_samples=max_train_samples,
                                        max_test_samples=max_test_samples,
                                        max_valid_samples=max_valid_samples)

    def reshape_dataset(self):
        """ """
        self.train_dataset = reshape_dataset(self.train_dataset)
        self.valid_dataset = reshape_dataset(self.valid_dataset)
        self.test_dataset = reshape_dataset(self.test_dataset)

    def one_hot_labels(self):
        """ """
        self.train_labels = one_hot_labels(self.train_labels)
        #print "train_labels shape=%s" % (self.train_labels.shape, )
        #print "train_labels=%s" % (self.train_labels, )
        self.valid_labels = one_hot_labels(self.valid_labels)
        self.test_labels = one_hot_labels(self.test_labels)

    @classmethod
    def _verify_data_is_balanced(cls, labels):
        """ """
        sorted_labels = sorted(labels)
        labels_count = {}

        for label in sorted_labels:
            if label not in labels_count:
                labels_count[label] = 1
            else:
                labels_count[label] += 1
        print "Verify that data is balanced across classes: %s" % (labels_count, )

    def verify_data_is_balanced(self):
        """ """
        self._verify_data_is_balanced(self.train_labels)
        self._verify_data_is_balanced(self.valid_labels)
        self._verify_data_is_balanced(self.test_labels)

    def show_random_sample(self):
        """ """
        mapIdxToChar = { 0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E',
                         5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}
        import matplotlib.pyplot as plt
        train_dataset_size = self.train_dataset.shape[0]
        random_index = random.choice(range(0, train_dataset_size))
        image_obj = plt.imshow(self.train_dataset[random_index, :, :])
        print "%s" % mapIdxToChar[self.train_labels[random_index]]
        plt.show()

    def __generate_pickle_file(self):
        """ """
        train_filename = maybe_download(url=DATA_URL, filename='notMNIST_large.tar.gz', dst_dir=DATA_DIR)
        test_filename = maybe_download(url=DATA_URL, filename='notMNIST_small.tar.gz', dst_dir=DATA_DIR)
        train_folders = maybe_extract(train_filename, DATA_DIR)
        test_folders = maybe_extract(test_filename, DATA_DIR)
        print test_folders
        display_random_image(test_folders)
        train_dataset, train_labels = load(data_folders=train_folders,
                                   image_size=28, pixel_depth=255.0,
                                   min_num_images=450000, max_num_images=550000)
        test_dataset, test_labels = load(data_folders=test_folders,
                                 image_size=28, pixel_depth=255.0,
                                 min_num_images=18000, max_num_images=20000)
        train_dataset, train_labels = randomize(train_dataset, train_labels)

        train_size = 200000
        valid_size = 10000

        valid_dataset = train_dataset[:valid_size, :, :]
        valid_labels = train_labels[:valid_size]
        train_dataset = train_dataset[valid_size:valid_size+train_size, :, :]
        train_labels = train_labels[valid_size:valid_size+train_size]
        print 'Training', train_dataset.shape, train_labels.shape
        print 'Validation', valid_dataset.shape, valid_labels.shape
        save_data(filename=self.pickle_file,
              train_dataset=train_dataset, train_labels=train_labels,
              valid_dataset=valid_dataset, valid_labels=valid_labels,
              test_dataset=test_dataset, test_labels=test_labels)



if  __name__ == "__main__":
    pickle_file = os.path.join(DATA_DIR, DATA_PICKLE_FILE)
    mnist = NotMNIST(pickle_file=pickle_file,
                     max_train_samples=5000,
                     max_valid_samples=500,
                     max_test_samples=500)
    for i in xrange(20):
        mnist.show_random_sample()
    sys.exit(0)

