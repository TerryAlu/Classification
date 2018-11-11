from datagen import *
from classifier import *

np.set_printoptions(suppress=True)

def csv_string(arr):
    """
    Convert 2D np array to csv string
    """
    assert len(arr.shape) == 2
    X, Y = arr.shape
    ret = ""
    for x in xrange(X):
        ret = ret + ",".join(arr[x].astype(str)) + "\n"

    return ret

def check_ttp(start, end, step, var, model, filename="data", linear=True, noise=False):
    t_tp = []
    # test 2D characteristic
    for x in xrange(start, end, step):
        # verbose("%s model: %d -> %d" % (model, x, end), color=INFO)
        t1 = Target("Target1", [(end,var)]*2, 100000)
        t2 = Target("Target2", [(x,var)]*2, 100000)
        dg = DataGenerator((t1, t2), 4, linear)
        dg.save(filename)
        cm = gen_cm_from_npz(filename, model, noise)
        # push (avg. tvalue, tp rate)
        t_tp.append([np.average(dg.tvalues), cm[0][0]])
    t_tp = np.array(t_tp)
    return t_tp

print "Experiment0: Show confusion matrix"
t1 = Target("Target1", [(150,20), (100,10), (60, 10)], 1000)
t2 = Target("Target2", [(145,20), (110,10), (75, 10)], 1000)
dg = DataGenerator((t1, t2), 4, False)
dg.save("data")
cm = gen_cm_from_npz("data", "DT")
verbose("Confusion Matrix", cm)

print "Experiment1: Verify relationship between t-values & TP rate"
ttp100_150 = check_ttp(100, 150, 2, 20, "DT", "data_dt100", linear=True)
ttp250_300 = check_ttp(250, 300, 1, 20, "DT", "data_dt250", linear=True)
verbose("TTP 100~150", ttp100_150)
print csv_string(ttp100_150)
verbose("TTP 250~300", ttp250_300)
print csv_string(ttp250_300)

print "Experiment2: Linear & Non-Linaer for DT"
ttp100_150_l = check_ttp(100, 150, 2, 20, "DT", "data_dt100_L", linear=True)
ttp100_150_nl = check_ttp(100, 150, 2, 20, "DT", "data_dt100_NL", linear=False)
verbose("TTP 100~150 linear", ttp100_150_l)
print csv_string(ttp100_150_l)
verbose("TTP 100~150 non-linear", ttp100_150_nl)
print csv_string(ttp100_150_nl)

print "Experiment3-1: Check influence of noise (10% noise)"
ttp100_150_01nnoise = check_ttp(100, 150, 2, 20, "DT", "data_dt100_01nnoise", noise=False)
ttp100_150_01noise = check_ttp(100, 150, 2, 20, "DT", "data_dt100_01noise", noise=0.1)
verbose("TTP 100~150 no noise", ttp100_150_01nnoise)
print csv_string(ttp100_150_01nnoise)
verbose("TTP 100~150 with 0.1 noise", ttp100_150_01noise)
print csv_string(ttp100_150_01noise)

print "Experiment3-2: Check influence of noise (30% noise)"
ttp100_150_03nnoise = check_ttp(100, 150, 2, 20, "DT", "data_dt100_03nnoise", noise=False)
ttp100_150_03noise = check_ttp(100, 150, 2, 20, "DT", "data_dt100_03noise", noise=0.3)
verbose("TTP 100~150 no noise", ttp100_150_03nnoise)
print csv_string(ttp100_150_03nnoise)
verbose("TTP 100~150 with 0.3 noise", ttp100_150_03noise)
print csv_string(ttp100_150_03noise)

print "Experiment3-3: Check influence of noise (50% noise)"
ttp100_150_05nnoise = check_ttp(100, 150, 2, 20, "DT", "data_dt100_05nnoise", noise=False)
ttp100_150_05noise = check_ttp(100, 150, 2, 20, "DT", "data_dt100_05noise", noise=0.5)
verbose("TTP 100~150 no noise", ttp100_150_05nnoise)
print csv_string(ttp100_150_05nnoise)
verbose("TTP 100~150 with 0.5 noise", ttp100_150_05noise)
print csv_string(ttp100_150_05noise)

print "Experiment4: Random Forest vs. Decision Tree (30% noise)"
ttp100_150_03noise_DT = check_ttp(100, 150, 2, 20, "DT", "data_dt100_03noise_DT", noise=0.5)
ttp100_150_03noise_RF = check_ttp(100, 150, 2, 20, "RF", "data_dt100_03noise_RF", noise=0.5)
verbose("TTP 100~150 DT with 0.3 noise", ttp100_150_03noise_DT)
print csv_string(ttp100_150_03noise_DT)
verbose("TTP 100~150 RF with 0.3 noise", ttp100_150_03noise_RF)
print csv_string(ttp100_150_03noise_RF)
