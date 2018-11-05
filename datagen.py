import numpy as np
from scipy import stats

vflag = True
HEADER = '\033[94m'
ENDC = '\033[0m'

def verbose(title, msg):
    if not vflag:
        return
    print "%s>>  %s  <<%s" % (HEADER, title, ENDC)
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
        self.constraints = [self.gen_constraint(t1) for _ in xrange(nnumeric)]

        # generate features
        self.t1data = [self.gen_feature(self.t1, constraint) for constraint in self.constraints]
        self.t1data = np.hstack(self.t1data)
        verbose("%s Features" % self.t1, self.t1data)
        self.t2data = [self.gen_feature(self.t2, constraint) for constraint in self.constraints]
        self.t2data = np.hstack(self.t2data)
        verbose("%s Features" % self.t2, self.t2data)

        # pvalues
        self.pvalues = self.get_pvalues()

    def gen_constraint(self, target):
        # 0~1000
        interception = 100*np.random.random()
        # 0~100
        coff = 10*np.random.random((target.char_len,))

        # verbose("Interception (0~1000)", interception)
        # verbose("Feature Coff. (0~100)", coff)

        if not self.linear:
            # power of characteristic (1~3)
            char_power = np.random.randint(1, 4, target.char_len)
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
        verbose("Chpow.", chpow)

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

    def get_pvalues(self):
        ncol = self.t1.char_len
        pvalues = []
        for i in xrange(ncol):
            t1col = self.t1data[...,i]
            t2col = self.t2data[...,i]
            t, p = stats.ttest_ind(t1col,t2col)
            pvalues.append(p)

        verbose("Pvalues", np.array(pvalues))

        return np.array(pvalues)

    def save(self, path, shuffle=True, csv=False):
        """
        Write data of two targets to file.
        If shuffle = True, then data of two targets will be mixed and shuffled before written to the file.
        Otherwise, t1 data will be written to file and followed by t2 data.
        """
        with open(path, "w") as fp:
            # write title to file
            title = ["ch"+str(x+1)for x in xrange(self.t1.char_len)]
            title.insert(0, "id")
            fp.write(",".join(title))
            fp.write("\n")

            # mix t1 & t2 data
            data = np.vstack([self.t1data, self.t2data])
            if shuffle:
                np.random.shuffle(data)
            
            # FIXME: This function takes too much time...
            if csv:
                for i in xrange(len(data)):
                    # write first column (id)
                    fp.write(str(i+1)+",")
                    # write others columns (characteristic)
                    data_str = data.astype(str)
                    fp.write(",".join(data_str[i]))
                    fp.write("\n")
            else:
                np.save(path, data)
        

if __name__ == "__main__":

    # XXX: Modify t1 & t2 characteristic parameter to conduct experiment
    t1 = Target("Target1", [(50,30), (100,50)], 10000)
    t2 = Target("Target2", [(45,20), (110,30)], 10000)
    dg = DataGenerator((t1, t2), 2, True)

    # dg.save("data.npy")
