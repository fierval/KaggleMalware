from pybrain.datasets            import ClassificationDataSet, UnsupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork

from pybrain.supervised.trainers import BackpropTrainer

from pybrain.structure.modules import SoftmaxLayer

from tr_utils import append_to_arr
from train_files import TrainFiles

import numpy as np
import time
from SupervisedLearning import SKSupervisedLearning

from sklearn.metrics import log_loss
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

import matplotlib.pyplot as plt

from sklearn.decomposition.pca import PCA


def _createDataSet(X, Y, one_based):
    labels = np.unique(Y)
    alldata = ClassificationDataSet(X.shape[1], nb_classes = labels.shape[0], class_labels = labels)
    shift = 1 if one_based else 0
    for i in range(X.shape[0]):
        alldata.addSample(X[i], Y[i] - shift)
    
    alldata._convertToOneOfMany()
    return alldata

def _createUnsupervisedDataSet(X):
    alldata = UnsupervisedDataSet(X.shape[1])
    for i in X:
        alldata.addSample(i)
    return alldata

def createDataSets(X_train, Y_train, X_test, Y_test, one_based = True):
    """
    Creates the data set. Handles one-based classifications (PyBrain uses zero-based ones).
    """
    trndata = _createDataSet(X_train, Y_train, one_based)
    tstdata = _createDataSet(X_test, Y_test, one_based)    
    return trndata, tstdata

def nn_log_loss(fnn, data):
    proba = fnn.activateOnDataset(data)
    return log_loss(data['target'], proba)

def train(trndata, tstdata, epochs = 100, test_error = 0.2, weight_decay = 0.0001, momentum = 0.5):
    """
    FF neural net
    """

    fnn = buildNetwork(trndata.indim, trndata.indim / 4, trndata.outdim, outclass = SoftmaxLayer)

    trainer = BackpropTrainer(fnn, trndata, momentum = momentum, weightdecay = weight_decay)

    epoch_delta = 1
    stop = False
    
    trnResults = np.array([])
    tstResults = np.array([])
    totEpochs = np.array([])

    trnLogLoss = np.array([])
    tstLogLoss = np.array([])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))
    #hold(True) # overplot on

    #plt.ion()
    while not stop:
        trainer.trainEpochs(epoch_delta)

        trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )
        tstresult = percentError( trainer.testOnClassData(
           dataset=tstdata ), tstdata['class'] )

        tstLogLoss = append_to_arr(tstLogLoss, nn_log_loss(fnn, tstdata))
        trnLogLoss = append_to_arr(trnLogLoss, nn_log_loss(fnn, trndata))

        print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          "  test error: %5.2f%%" % tstresult, \
          " test logloss: %2.4f" % tstLogLoss[-1], \
          " train logloss: %2.4f" % trnLogLoss[-1]

        
        trnResults = append_to_arr(trnResults, trnresult)
        tstResults = append_to_arr(tstResults, tstresult)
        totEpochs = append_to_arr(totEpochs, trainer.totalepochs)
        
        plt.sca(ax1)
        plt.cla()
        ax1.plot(totEpochs, trnResults, label = 'Train')
        ax1.plot(totEpochs, tstResults, label = 'Test')

        plt.sca(ax2)
        plt.cla()
        ax2.plot(totEpochs, trnLogLoss, label = 'Train')
        ax2.plot(totEpochs, tstLogLoss, label = 'Test')

        ax1.legend()
        ax2.legend()

        plt.draw()
        time.sleep(0.1)
        plt.pause(0.0001)

        stop = (tstLogLoss[-1] <= test_error or trainer.totalepochs >= epochs)
    return fnn

def predict_nn(trndata, epochs = 300, test_error = 0.0147, weight_decay = 0.0001, momentum = 0.15):
    """
    FF neural net
    """

    fnn = buildNetwork(trndata.indim, trndata.indim / 4, trndata.outdim, outclass = SoftmaxLayer)

    trainer = BackpropTrainer(fnn, trndata, momentum = momentum, weightdecay = weight_decay)

    epoch_delta = 1
    stop = False

    totEpochs = np.array([])
    trnResults = np.array([])
    trnLogLoss = np.array([])

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,8))

    while not stop:
        trainer.trainEpochs(epoch_delta)

        trnresult = percentError( trainer.testOnClassData(),
                              trndata['class'] )

        trnLogLoss = append_to_arr(trnLogLoss, nn_log_loss(fnn, trndata))

        print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult, \
          " train logloss: %2.4f" % trnLogLoss[-1]

        
        trnResults = append_to_arr(trnResults, trnresult)
        totEpochs = append_to_arr(totEpochs, trainer.totalepochs)

        plt.sca(ax1)
        plt.cla()
        ax1.plot(totEpochs, trnResults, label = 'Train')

        plt.sca(ax2)
        plt.cla()
        ax2.plot(totEpochs, trnLogLoss, label = 'Train')

        ax1.legend()
        ax2.legend()

        plt.draw()
        time.sleep(0.1)
        plt.pause(0.0001)

        stop = (trnLogLoss[-1] <= test_error or trainer.totalepochs >= epochs)
    return fnn

train_path_mix = "/kaggle/malware/train/mix_lbp"
train_path_freq = "/kaggle/malware/train/instr_freq"
labels_file = "/kaggle/malware/trainLabels.csv"
csv_file = "/kaggle/malware/mix_lbp.csv"

def do_train():
    X, Y, Xt, Yt = TrainFiles.from_csv(csv_file)
    sl = SKSupervisedLearning(SVC, X, Y, Xt, Yt)
    sl.fit_standard_scaler()

    #pca = PCA(250)
    #pca.fit(np.r_[sl.X_train_scaled, sl.X_test_scaled])
    #X_pca = pca.transform(sl.X_train_scaled)
    #X_pca_test = pca.transform(sl.X_test_scaled)
    
    ##construct a dataset for RBM
    #X_rbm = X[:, 257:]
    #Xt_rbm = X[:, 257:]

    #rng = np.random.RandomState(123)
    #rbm = RBM(X_rbm, n_visible=X_rbm.shape[1], n_hidden=X_rbm.shape[1]/4, numpy_rng=rng)

    #pretrain_lr = 0.1
    #k = 2
    #pretraining_epochs = 200
    #for epoch in xrange(pretraining_epochs):
    #    rbm.contrastive_divergence(lr=pretrain_lr, k=k)
    #    cost = rbm.get_reconstruction_cross_entropy()
    #    print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost


    trndata, tstdata = createDataSets(X, Y, X_test, Yt)
    fnn = train(trndata, tstdata, epochs = 1000, test_error = 0.025, momentum = 0.15, weight_decay = 0.0001)

def do_predict(X_train, Y_train, X_test):
    trndata = _createDataSet(X_train, Y_train, one_based = True)
    tstdata = _createUnsupervisedDataSet(X_test)
    fnn = predict_nn(trndata)
    proba = fnn.activateOnDataset(tstdata)

def do_train_with_freq():
    tf_mix = TrainFiles(train_path = train_path_mix, labels_file = labels_file, test_size = 0.)
    tf_freq = TrainFiles(train_path = train_path_freq, labels_file = labels_file, test_size = 0.)

    X_m, Y_m, _, _ = tf_mix.prepare_inputs()
    X_f, Y_f, _, _ = tf_freq.prepare_inputs()

    X = np.c_[X_m, X_f]
    Y = Y_f

    X, Xt, Y, Yt = train_test_split(X, Y, test_size = 0.1)
    sl = SKSupervisedLearning(SVC, X, Y, Xt, Yt)
    sl.fit_standard_scaler()

    pca = PCA(250)
    pca.fit(np.r_[sl.X_train_scaled, sl.X_test_scaled])
    X_pca = pca.transform(sl.X_train_scaled)
    X_pca_test = pca.transform(sl.X_test_scaled)

    #sl.train_params = {'C': 100, 'gamma': 0.0001, 'probability' : True}
    #print "Start SVM: ", time_now_str()
    #sl_ll_trn, sl_ll_tst = sl.fit_and_validate()
    #print "Finish Svm: ", time_now_str()

    ##construct a dataset for RBM
    #X_rbm = X[:, 257:]
    #Xt_rbm = X[:, 257:]

    #rng = np.random.RandomState(123)
    #rbm = RBM(X_rbm, n_visible=X_rbm.shape[1], n_hidden=X_rbm.shape[1]/4, numpy_rng=rng)

    #pretrain_lr = 0.1
    #k = 2
    #pretraining_epochs = 200
    #for epoch in xrange(pretraining_epochs):
    #    rbm.contrastive_divergence(lr=pretrain_lr, k=k)
    #    cost = rbm.get_reconstruction_cross_entropy()
    #    print >> sys.stderr, 'Training epoch %d, cost is ' % epoch, cost


    trndata, tstdata = createDataSets(X_pca, Y, X_pca_test, Yt)
    fnn = train(trndata, tstdata, epochs = 1000, test_error = 0.025, momentum = 0.2, weight_decay = 0.0001)
