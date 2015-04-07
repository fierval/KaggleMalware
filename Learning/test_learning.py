from tr_utils import time_now_str

import sklearn.svm as svm
import numpy as np

from train_files import TrainFiles
from SupervisedLearning import SKSupervisedLearning
from sklearn.decomposition import PCA

out_labels = "/kaggle/malware/scratchpad/submission_fake.csv"
# instantiate file system interface
tf = TrainFiles('/kaggle/malware/scratchpad/train/1dlbp', '/kaggle/malware/scratchpad/test/1dlbp', "/kaggle/malware/trainLabels.csv")

# read in our data
X_train, Y_train, X_test, Y_test = tf.prepare_inputs()

sl = SKSupervisedLearning(svm.SVC, X_train, Y_train, X_test, Y_test)
sl.fit_standard_scaler()


def test_fit_svm():

    # start fitting
    sl.train_params = {'probability': True, 'C': 100, 'gamma': 0.1}

    print "Starting: ", time_now_str()
    # logloss is the score
    tscore, valscore = sl.fit_and_validate()
    print "Finished: ", time_now_str()
    print "Train log loss: {0}, test log loss: {1}".format(tscore, valscore)

def test_grid_svm():
    sl.train_params = {'probability': True}
    sl.cv = 3

    sl.estimation_params = {'C': [1., 100.], 'gamma': [0.1, 0.01]}
    sl.grid_search_classifier()

