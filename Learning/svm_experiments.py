from SupervisedLearning import SKSupervisedLearning
from train_files import TrainFiles
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import log_loss, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from tr_utils import vote
import matplotlib.pylab as plt
from train_nn import createDataSets, train

train_path_mix = "/kaggle/malware/mix_lbp.csv"
labels_file = "/kaggle/malware/trainLabels.csv"

X, Y_train, Xt, Y_test = TrainFiles.from_csv(train_path_mix)

def plot_confusion(sl):
    conf_mat = confusion_matrix(sl.Y_test, sl.clf.predict(sl.X_test_scaled)).astype(dtype='float')
    norm_conf_mat = conf_mat / conf_mat.sum(axis = 1)[:, None]

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(norm_conf_mat, cmap=plt.cm.jet, 
                    interpolation='nearest')
    cb = fig.colorbar(res)
    labs = np.unique(Y_test)
    x = labs - 1

    plt.xticks(x, labs)
    plt.yticks(x, labs)

    for i in x:
        for j in x:
            ax.text(i - 0.2, j + 0.2, "{:3.0f}".format(norm_conf_mat[j, i] * 100.))
    return conf_mat

sl = SKSupervisedLearning(SVC, X, Y_train, Xt, Y_test)
sl.fit_standard_scaler()
sl.train_params = {'C': 100, 'gamma': 0.01, 'probability' : True}
ll_trn, ll_tst = sl.fit_and_validate()

print "SVC log loss: ", ll_tst

conf_svm = plot_confusion(sl)

#Neural net
trndata, tstdata = createDataSets(sl.X_train_scaled, Y_train, sl.X_test_scaled, Y_test)
fnn = train(trndata, tstdata, epochs = 1000, test_error = 0.025, momentum = 0.15, weight_decay = 0.0001)

sl_ccrf = SKSupervisedLearning(CalibratedClassifierCV, X, Y_train, Xt, Y_test)
sl_ccrf.train_params = \
    {'base_estimator': RandomForestClassifier(**{'n_estimators' : 7500, 'max_depth' : 200}), 'cv': 10}
sl_ccrf.fit_standard_scaler()
ll_ccrf_trn, ll_ccrf_tst = sl_ccrf.fit_and_validate()

print "Calibrated log loss: ", ll_ccrf_tst
conf_ccrf = plot_confusion(sl_ccrf)

#predicted = cross_val_predict(SVC(**sl.train_params), sl.X_train_scaled, n_jobs = -1, y = Y_train, cv=10)

#fig,ax = plt.subplots()
#ax.scatter(Y_train, predicted)
#ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#ax.xticks(np.unique(Y_test))
#ax.yticks(np.unique(Y_test))

#fig.show()

x = 1. / np.arange(1., 6)
y = 1 - x

xx, yy = np.meshgrid(x, y)
lls1 = np.zeros(xx.shape[0] * yy.shape[0]).reshape(xx.shape[0], yy.shape[0])
lls2 = np.zeros(xx.shape[0] * yy.shape[0]).reshape(xx.shape[0], yy.shape[0])

for i, x_ in enumerate(x):
    for j, y_ in enumerate(y):
        proba = vote([sl.proba_test, sl_ccrf.proba_test], [x_, y_])
        lls1[i, j] = log_loss(Y_test, proba)

        proba = vote([sl.proba_test, sl_ccrf.proba_test], [y_, x_])
        lls2[i, j] = log_loss(Y_test, proba)

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(121)
ax1 = fig.add_subplot(122)

ax.set_aspect(1)
ax1.set_aspect(1)

res = ax.imshow(np.array(lls1), cmap=plt.cm.jet, 
                interpolation='nearest')
res = ax1.imshow(np.array(lls2), cmap=plt.cm.jet, 
                interpolation='nearest')

cb = fig.colorbar(res)

