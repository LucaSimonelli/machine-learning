import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets
import cPickle as pickle
from process_data import randomize, load_data
import sys

pickle_file = "./data/notMNIST.pickle"
(X, Y, _, _, Zx, Zy) = load_data(pickle_file=pickle_file,
                                 max_train_samples=5000,
                                 max_valid_samples=0,
                                 max_test_samples=500,
                                 one_hot_labels=False)
print X.shape
print Y.shape
#sys.exit(0)
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
