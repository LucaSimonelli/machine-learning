from sklearn import linear_model
import os
from process_data import (NotMNIST,
                          DATA_DIR, DATA_PICKLE_FILE, DATA_IMAGE_SIZE,
                          DATA_NUM_LABELS)


MNIST = NotMNIST(pickle_file=os.path.join(DATA_DIR, DATA_PICKLE_FILE),
                 max_train_samples=50000,
                 max_valid_samples=0,
                 max_test_samples=5000)

MNIST.verify_data_is_balanced()
MNIST.reshape_dataset()

X = MNIST.train_dataset
Y = MNIST.train_labels
Zx = MNIST.test_dataset
Zy = MNIST.test_labels
print X.shape
print Y.shape
logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

predicted_labels = logreg.predict(Zx)
wrong_predictions = 0.0
right_predictions = 0.0
for pair in zip(Zy, predicted_labels):
    if pair[0] != pair[1]:
        wrong_predictions += 1.0
    else:
        right_predictions += 1.0
print "Right=%f Wrong=%f Total=%f Accuracy=%f" %(right_predictions,
    wrong_predictions, right_predictions+wrong_predictions,
    right_predictions / (right_predictions+wrong_predictions) )
