from tr_utils import time_now_str

import sklearn.svm as svm
import numpy as np

from train_files import TrainFiles
from SupervisedLearning import SKSupervisedLearning
from sklearn.decomposition import SparsePCA, PCA
from train_nn import train, createDataSets
from sklearn.naive_bayes import MultinomialNB
from sklearn.lda import LDA
from sklearn.preprocessing import normalize

tf = TrainFiles('/kaggle/malware/scratchpad/text/train/instr_freq', '/kaggle/malware/scratchpad/text/test/instr_freq', "/kaggle/malware/trainLabels.csv")
tf1 = TrainFiles('/kaggle/malware/scratchpad/train/1dlbp', '/kaggle/malware/scratchpad/test/1dlbp', "/kaggle/malware/trainLabels.csv")

X_train, Y_train, X_test, Y_test = tf1.prepare_inputs()

n_components = 300
pca = PCA(n_components = n_components)
pca.fit(np.r_[X_train, X_test])

#n_components = np.where(np.cumsum(pca.explained_variance_ratio_) >= 0.99)[0][0]
#print n_components

#pca = PCA(n_components = n_components)
#pca.fit(np.r_[X_train, X_test])
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# Naive Bayes

sl = SKSupervisedLearning(LDA, X_train, Y_train, X_test, Y_test)
#sl.fit_standard_scaler()
trndata, tstdata = createDataSets(normalize(X_train), Y_train, normalize(X_test), Y_test)
train(trndata, tstdata, epochs = 1000, weight_decay = 0.0001, momentum = 0.15)

ll = sl.fit_and_validate()

print "Log loss: ", ll