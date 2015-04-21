from tr_utils import write_to_csv, time_now_str, vote_reduce, vote
from SupervisedLearning import SKSupervisedLearning
from train_files import TrainFiles
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from os import path
from sklearn.metrics import log_loss
from sklearn.decomposition.pca import PCA
from sklearn.calibration import CalibratedClassifierCV

from train_nn import predict_nn, _createDataSet, _createUnsupervisedDataSet

prediction = True
doTrees = True

def predict():
    tf = TrainFiles('/kaggle/malware/train/mix_lbp', val_path = '/kaggle/malware/test/mix_lbp', labels_file = "/kaggle/malware/trainLabels.csv")

    X_train, Y_train, X_test, Y_test = tf.prepare_inputs()

    sl_svm = SKSupervisedLearning(SVC, X_train, Y_train, X_test, Y_test)
    sl_svm.fit_standard_scaler()
    sl_svm.train_params = {'C': 100, 'gamma': 0.01, 'probability': True}

    print "Starting SVM: ", time_now_str()
    _, ll_svm = sl_svm.fit_and_validate()

    print "SVM score: {0:.4f}".format(ll_svm if not prediction else _)
    print "Finished training SVM: ", time_now_str()

    # neural net
    print "Starting NN: ", time_now_str()

    trndata = _createDataSet(sl_svm.X_train_scaled, Y_train, one_based = True)
    tstdata = _createUnsupervisedDataSet(sl_svm.X_test_scaled)
    fnn = predict_nn(trndata)
    proba_nn = fnn.activateOnDataset(tstdata)

    print "Finished training NN: ", time_now_str()

    # no validation labels on actual prediction
    if doTrees:
        # random forest
        sl_ccrf = SKSupervisedLearning(CalibratedClassifierCV, X_train, Y_train, X_test, Y_test)
        sl_ccrf.train_params = \
            {'base_estimator': RandomForestClassifier(**{'n_estimators' : 7500, 'max_depth' : 200}), 'cv': 10}
        sl_ccrf.fit_standard_scaler()

        print "Starting on RF: ", time_now_str()
        ll_ccrf_trn, ll_ccrf_tst = sl_ccrf.fit_and_validate()

        print "RF score: {0:.4f}".format(ll_ccrf_tst if not prediction else ll_ccrf_trn)
        sl_ccrf.proba_test.tofile("/temp/sl_ccrf.prob")
        sl_svm.proba_test.tofile("/temp/sl_svm.prob")
        proba_nn.tofile("/temp/nn.prob")

        print "Finished training RF: ", time_now_str()

    if prediction:
        proba = vote([sl_svm.proba_test, sl_ccrf.proba_test, proba_nn], [2./3., 1./6., 1./3.])

        out_labels = "/kaggle/malware/submission33.csv"
        task_labels = "/kaggle/malware/testLabels.csv"
        labels = [path.splitext(t)[0] for t in tf.get_val_inputs()]
        out = write_to_csv(task_labels, labels, proba, out_labels)

    else:
        # visualize the decision surface, projected down to the first
        # two principal components of the dataset
        pca = PCA(n_components=2).fit(sl_svm.X_train_scaled)

        X = pca.transform(sl_svm.X_train_scaled)

        x = np.arange(X[:, 0].min() - 1, X[:, 1].max() + 1, 1)
        y = np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 1)

        xx, yy = np.meshgrid(x, y)

        # title for the plots
        titles = ['SVC with rbf kernel',
                  'Random Forest \n'
                  'n_components=7500',
                  'Decision Trees \n'
                  'n_components=7500']

        #plt.tight_layout()
        plt.figure(figsize=(12, 5))

        # predict and plot
        for i, clf in enumerate((sl_svm.clf, sl_rfc.clf, sl_trees.clf)):
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, m_max]x[y_min, y_max].
            plt.subplot(1, 3, i + 1)
            clf.fit(X, Y_train)
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)
            plt.axis('off')

            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=Y_train, cmap=plt.cm.Paired)

            plt.title(titles[i])
        plt.tight_layout()
        plt.show()

def write_canned_predictions(tf, task_labels, out_labels):
    proba_svm = np.fromfile('/kaggle/malware/results/sl_svm.prob').reshape(-1, 9, order = 'C')
    proba_rf = np.fromfile('/kaggle/malware/results/sl_ccrf.prob').reshape(-1, 9, order = 'C')

    proba = vote([proba_svm, proba_rf], [4./5., 1./3.])

    labels = [path.splitext(t)[0] for t in tf.get_val_inputs()]
    out = write_to_csv(task_labels, labels, proba, out_labels)
