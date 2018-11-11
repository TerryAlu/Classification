import numpy as np
from scipy import stats

vflag = True
HEADER = '\033[94m'
OKGREEN = '\033[92m'
ENDC = '\033[0m'

def verbose(title, msg=None):
    if not vflag:
        return
    print "%s>>  %s  <<%s" % (HEADER, title, ENDC)
    if msg is not None:
        print msg
    print ""

class Target():
    """
    Target classes wapper.

    @param  characteristic    [(mu1, sigma1), (mu2, sigma2), ....]
    """
    def __init__(self, name, characteristic, size=100000):
        assert name and characteristic and size > 0

        self.name = name
        self.size = size
        self.char = characteristic
        self.char_list = self.gen_char(characteristic, size)
        self.char_len = len(characteristic)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def gen_char(self, characteristic, size):
        char_list = [np.array([np.random.normal(mu, sigma, size)]).T for mu, sigma in characteristic]
        char_list = np.hstack(char_list)
        return char_list

class DataGenerator():
    def __init__(self, targets, nnumeric, linear=True):
        """
        @param  targets          (Target1, Target2)
        @param  nnumeric         # of numeric features

        @attr   t1data           Each datum contains all features of the target
                                 [[<numeric1>, <numeric2>...]]

                t2data

                constraints      f(w) = interception + a*char1^pow1 + b*char2^pow2 + ... 
                                 [interception, (a, pow1), (b, pow2)...]

                pvalues          pvalues between each features of targets
                                 [p1, p2, p3...]
                                 
        """
        self.t1, self.t2 = targets
        assert self.t1 and isinstance(self.t1, Target)
        assert self.t2 and isinstance(self.t2, Target)
        assert self.t1.char_len == self.t2.char_len

        self.nnumeric = nnumeric
        assert self.nnumeric > 0

        self.linear = linear

        # number of constraint = number of features = nnumeric
        self.constraints = [self.gen_constraint(self.t1) for _ in xrange(nnumeric)]

        # generate features
        self.t1data = [self.gen_feature(self.t1, constraint) for constraint in self.constraints]
        self.t1data = np.hstack(self.t1data)
        # verbose("%s Features" % self.t1, self.t1data)
        self.t2data = [self.gen_feature(self.t2, constraint) for constraint in self.constraints]
        self.t2data = np.hstack(self.t2data)
        # verbose("%s Features" % self.t2, self.t2data)

        # tvalues & pvalues
        self.tvalues, self.pvalues = self.ttest()

    def gen_constraint(self, target):
        # 0~1000
        interception = 100*np.random.random()
        # 0~100
        coff = 10*np.random.random((target.char_len,))

        # verbose("Interception (0~1000)", interception)
        # verbose("Feature Coff. (0~100)", coff)

        if not self.linear:
            # power of characteristic (2~5)
            char_power = np.random.randint(2, 5, target.char_len)
        else:
            char_power = np.ones(target.char_len)

        # verbose("Power of Characteristic", char_power)

        res = []
        for i in xrange(target.char_len):
            res.append((coff[i], char_power[i]))
        res.insert(0, interception)

        # verbose("Constraint of %s" % target, res)
        return res

    def gen_feature(self, target, constraint):
        """
        Generate a feature by a constraint function
        f(w) = interception + a*char1^pow1 + b*char2^pow2 + ... 
        """

        cnp = np.array(constraint[1:])

        # extract variables
        interception = constraint[0]
        coff = cnp[...,0]
        chpow = cnp[...,1]

        # verbose("Interception", interception)
        # verbose("Coff.", coff)
        # verbose("Chpow.", chpow)

        # [char^power]
        power_chars = np.power(target.char_list, chpow)
        # [coff*char^power]
        coff_chars = np.multiply(power_chars, coff)
        # [result value]
        res = np.sum(coff_chars, axis=1)
        res = res + interception

        # verbose("Characteristic", target.char_list)
        # verbose("Power", power_chars)
        # verbose("Coff", coff_chars)

        res = np.array([res]).T
        # verbose("Result", res)

        return res

    def ttest(self):
        pvalues = []
        tvalues = []
        for i in xrange(self.nnumeric):
            t1col = self.t1data[...,i]
            t2col = self.t2data[...,i]
            t, p = stats.ttest_ind(t1col,t2col)
            tvalues.append(t)
            pvalues.append(p)

        # verbose("Tvalues", np.array(tvalues))
        # verbose("Pvalues", np.array(pvalues))

        return np.array(tvalues), np.array(pvalues)

    def save(self, path):
        """
        Write data of two targets to file.
        """
        # mix t1 & t2 data
        np.savez(path, t1=self.t1data, t2=self.t2data)
        

if __name__ == "__main__":

    # XXX: Modify t1 & t2 characteristic parameter to conduct experiment
    # t1 = Target("Target1", [(50,30), (100,50)], 10000)
    # t2 = Target("Target2", [(45,20), (110,30)], 10000)

    t1 = Target("Target1", [(150,20), (100,10)], 100000)
    t2 = Target("Target2", [(145,20), (110,10)], 100000)

    dg = DataGenerator((t1, t2), 2, True)
    dg.save("data")
