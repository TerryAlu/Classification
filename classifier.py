import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
# models
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

def related_shuffle(data, cls):
    """
    Shuffle data and classification arrays to the same order
    """
    s = np.arange(data.shape[0])
    np.random.shuffle(s)
    return data[s], cls[s]

vflag = True
HEADER = '\033[94m'
INFO = '\033[92m'
ENDC = '\033[0m'

def verbose(title, msg=None, color=HEADER):
    if not vflag:
        return
    print "%s>>  %s  <<%s" % (color, title, ENDC)
    if msg is not None:
        print msg
    print ""

def gen_cm_from_npz(path, model="DT", noise=False):
    """
    Generate confusion matrix corresponding to data in npz file
    75% data for training, 25% data for testing
    Noise is a random number between 0.5*min and 1.5*max of the column

    @param  model  DecisionTree (DT) | RandomForest (RF) | GaussianNB (GNB) | LinearSVM (LSVM)
    @param  noise  Percentage of noise
    """
    data = np.load(path+".npz")
    t1 = data['t1']
    t2 = data['t2']

    mixdata = np.vstack([t1, t2])
    mixclass = np.hstack([np.zeros(len(t1)), np.ones(len(t2))])

    # add noise
    if noise is not False and (0 <= noise and noise < 1):
        noise_data = []
        row_n, col_n = mixdata.shape
        noise_num = (int)(1.0*row_n*noise)

        # generate noise data
        for i in xrange(col_n):
            mini = np.amin(mixdata[...,i])
            maxi = np.amax(mixdata[...,i])
            noise_data.append(np.random.randint((int)(0.5*mini), (int)(1.5*maxi), noise_num))
        noise_data = np.array(noise_data).T
        # generate noise class
        noise_class = np.random.randint(0, 2, noise_num)

        # merge noise to original data
        mixdata = np.vstack([mixdata, noise_data])
        mixclass = np.hstack([mixclass, noise_class])

    # shuffle and split train & test data
    data_train, data_test, cls_train, cls_test = train_test_split(mixdata, mixclass, test_size=0.25)

    if model in ["DT", "DecisionTree"]:
        clf = dtc()
    elif model in ["RF", "RandomForest"]:
        clf = RandomForestClassifier(n_estimators=10)
    elif model in ["GNB", "GaussianNB"]:
        clf = GaussianNB()
    elif model in ["LSVM", "LinearSVM"]:
        clf = svm.SVC(kernel="linear")
    else:
        raise Exception("Unknown Model %s" % model)

    # verbose("Train %s model" % model, color=INFO)
    clf.fit(data_train, cls_train)
    # verbose("Test %s model" % model, color=INFO)
    cls_pred = clf.predict(data_test)
    cm = confusion_matrix(cls_test, cls_pred)
    cm = 1.0*cm/len(cls_test)

    # verbose("Confusion Matrix", cm)

    return cm

if __file__ == "__main__":
    gen_cm_from_npz("data.npz")

