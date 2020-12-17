from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from functools import partial
from scipy.stats import entropy
import numpy as np
import pandas as pd
import math
import time

from pdb import set_trace as st

columns = ['p', 'X', 'Y', 'Z']


def set_product(a, b):
    return len(set(a).intersection(b))
    

def parallel_set_kronecker(A, B, n_jobs=4, backend="threads"):
    """
     Computes set_kronecker product A (x) B using multiple CPUs, with A and B
     being arrays or lists of sets.

     'backend' in {"threads", "processes"}
     
    """
    # this iterator returns the functions to execute for each task
    result = Parallel(n_jobs=n_jobs, prefer=backend) (
        delayed(set_product)(*j)
            for j in itertools.product(A, B)
    )
    # merging the output of the jobs
    return np.reshape(result, (len(A), len(B)))


class SetHashedDict:
    
    def __init__(self):
        self.x_keys = []
        self.y_keys = []
        self.values = {}
        self.x_memory = {}
        self.y_memory = {}


    def __getitem__(self, ks):
        if isinstance(ks, list) or isinstance(ks, tuple):
            ka, kb = ks
            ix = self.x_keys.index(ka)
            iy = self.y_keys.index(kb)

            return self.values[(ix, iy)]
        else:
            ix = self.x_keys.index(ks)
            return self.values[ix]
        

    def __setitem__(self, ks, value):
        if isinstance(ks, list) or isinstance(ks, tuple):
            ka, kb = ks
            if not ka in self.x_keys:
                self.x_keys.append(ka)

            if not kb in self.y_keys:
                self.y_keys.append(kb)
        
            self.values[(
                self.x_keys.index(ka),
                self.y_keys.index(kb)
            )] = value
        else:
            if not ks in self.x_keys:
                self.x_keys.append(ks)
            
            self.values[self.x_keys.index(ks)] = value
        

    def __contains__(self, ks):
        if isinstance(ks, list) or isinstance(ks, tuple):
            ka, kb = ks
            return ka in self.x_keys and kb in self.y_keys
        else:
            return ks in self.x_keys


    def keys(self):
        if len(self.y_keys) > 0:
             for x, y in zip(self.x_keys, self.y_keys):
                yield x, y
        else:
            for k in self.x_keys:
                yield k

    def items(self):
        if len(self.y_keys) > 0:
            for it in self.values.items():
                yield (
                    self.x_keys[it[0][0]],
                    self.y_keys[it[0][1]], it[1])
        else:
            for it in self.values.items():
                yield (self.x_keys[it[0]], it[1])


    def set_coo_mem(self, set_lk):
        it, lk = set_lk
        try:
            idx = self.x_keys.index(it)
            self.x_memory[idx] = lk

        except ValueError:
            print("such item {} isn't already in self.x_keys".format(it))
        
        
    def get_coo_mem(self, it):
        try:
            idx = self.x_keys.index(it)
            return self.x_memory[idx]

        except ValueError:
            print("such item {} isn't already in self.x_keys".format(it))
            


class RandomSetDistributions(object):

    def __init__(
            self, gamma=None, joint_method='conditional', exp=False, n_jobs=-1,
            backend='processes', ngram_range=(1,4), analyzer='char_wb'):
        #self.density = density
        self.gamma = gamma
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.xy_lkhoods = []
        self.x_lkhoods = []
        self.xy_logits = {}
        self.x_logits = {}
        self.joint_method = joint_method
        self.n_jobs = n_jobs
        self.backend=backend
        self.exp = exp
        self.set_rvs = {}
        self.prob_distributions = {}
        self.it_metrics = {}
        self.Omega = {}



    def _joint_pmf(self, logit):
#        self.gamma = (self.gamma if not self.gamma is None
#                                           else 1.0 / np.mean(logits))
#        return math.exp(self.gamma * sum(logits))
        try:
            return math.exp(self.gamma * logit)
        except TypeError:
            print("First define 'gamma' or compute conditional distribution "
                    "before using 'self._joint_pmf()'.")


    def _conditional_pmf(self, logits):
        self.gamma = (self.gamma if not self.gamma is None
                            else np.mean(logits)/np.var(logits) ** 2)
                            # Tom Mitchell (2017) for Gaussian priors
                            
        #return 1.0 / (1 + math.exp(-self.gamma * sum(logits)))
        return math.exp(self.gamma * sum(logits))


    def _total_prob_law(self, x, Y, mem):
        lkhood = 1.0
        #lkhoods = []
        #partition = 0.0
        if not x in mem.x_keys:
            for y in Y:
                intersect = x.intersection(y)
                mem[x, y] = {
                        'intersect': intersect,
                        'pmf': self._joint_pmf(len(intersect))
                }
                lkhood *= mem[x, y]['pmf']
        
                self.jpartition += mem[x, y]['pmf']
                #partition += mem[x, y]['pmf']
            mem.set_coo_mem((x, lkhood))
        else:
            lkhood = 0.0
        #lkhoods.append(lkhood)
        return lkhood  #s, partition
            

    def _fxy(self, omegaX, omegaY, x_, y_):
        if self.exp:
            return self._joint_pmf(
                len(omegaX.intersection(x_)) * len(omegaY.intersection(y_))
            )
        else:
            return len(omegaX.intersection(x_)) * len(omegaY.intersection(y_))
            

    def _cond_entropy(self, rvs=['X', 'Y']):
        P_XgY = self.prob_distributions['P_' + '|'.join(rvs)]
        P_Y = self.prob_distributions['P_' + rvs[1]]
        omega_x = list(self.prob_distributions['P_' + rvs[0]].keys())
        omega_y = list(self.prob_distributions['P_' + rvs[1]].keys())
        
        H_Xgy = []
        for y in omega_y:
            H_Xgy.append(P_Y[y] * entropy([P_XgY[x, y] for x in omega_x]))
            
        H_XgY = sum(H_Xgy)
        self.it_metrics['H(' + '|'.join(rvs) + ')'] = H_XgY
        
        return H_XgY


    def _entropy(self, rv):
        P_X = self.prob_distributions['P_' + rv]
        omega_x = list(self.prob_distributions['P_' + rv].keys())
        H_X = entropy([P_X[x] for x in omega_x], base=2)
        self.it_metrics['H(' + rv + ')'] = H_X
        
        return H_X
        
        
    def _mutual_info(self, rvs=['X', 'Y']):
        I_XY = self._entropy(rvs[0]) - self._cond_entropy(rvs)
        
        return I_XY


    def _marginal(self, rv='X'):
        assert isinstance(rv, str)

        try:
            self.Omega[rv] = self._build_vocab(rv)
            ints_x = self._build_itersec(rv)
        except:
            self.set_rvs[rv] = self._build_set_outcomes(rv)
            self.Omega[rv] = self._build_vocab(rv)
            ints_x = self._build_itersec(rv)

        
        mem = SetHashedDict()
        partition = 0
        for x in ints_x.keys():
            f_X = sum(ints_x[x])
            mem[x] = f_X
            partition += f_X
        mem.set_coo_mem((x, partition))

        distribution = SetHashedDict()
        for x in self.Omega[rv]:
            distribution[x] = mem[x] / partition
            
        self.prob_distributions['P_' + rv] = distribution
        self.xy_logits[rv] = mem


    def _build_set_outcomes(self, rv):
        analyzer = CountVectorizer(
            analyzer=self.analyzer, ngram_range=self.ngram_range)
        tokenizer = analyzer.build_analyzer()
        sets = []
        for x in self.df[rv]:
            try:
                sets.append(set(tokenizer(x)))
            except:
                print("ERROR {}".format(x))
                exit()
        #return [set(tokenizer(x)) for x in self.df[rv]]
        return sets

    def _conditional_(self, rvs=['X', 'Y']):
        assert (isinstance(rvs, list) or
                isinstance(rvs, tuple) and
                len(rvs) == 2), "Check 'rvs' argument for two RVs"

        mem = SetHashedDict()
        
        for x in self.set_rvs[rvs[0]]:
            partition = 0
            if x in mem.x_keys:
                continue
            else:
                for y in self.set_rvs[rvs[1]]:
                    f_xy = self._fxy(omega_X, omega_Y, x, y)
                    mem[x, y] = f_xy
                    partition += f_xy
                    
                mem.set_coo_mem((x, partition))
                
        P_XgY = [
            mem[x, y] / mem.get_coo_mem(x)
                for x, y in zip(self.set_rvs[rvs[0]], self.set_rvs[rvs[1]])
        ]   
        self.xy_logits['_'.join(rvs)] = mem
        self.df['P_' + '|'.join(rvs)] = P_XgY
        

    def _build_vocab(self, rv):
        omega = []
        for x in self.set_rvs[rv]:
            if x in omega:
                continue
            else:
                omega.append(x)
                
        return omega
        
        
    def _build_itersec(self, rv):
        ints_rv = SetHashedDict()
        for x in self.Omega[rv]:
            ints = []
            for x_ in self.set_rvs[rv]:
                ints.append(len(x.intersection(x_)))
            ints_rv[x] = ints
            
        return ints_rv
                       
                
    def _conditional(self, rvs=['X', 'Y']):
        assert (isinstance(rvs, list) or
                isinstance(rvs, tuple) and
                len(rvs) == 2), "Check 'rvs' argument for two RVs"
        
        try:
            omega_x = self.Omega[rvs[0]]
            omega_y = self.Omega[rvs[1]]
        except KeyError:
            self.Omega[rvs[0]] = self._build_vocab(rvs[0])
            self.Omega[rvs[1]] = self._build_vocab(rvs[1])
            omega_x = self.Omega[rvs[0]]
            omega_y = self.Omega[rvs[1]]

        ints_x = self._build_itersec(rvs[0])
        ints_y = self._build_itersec(rvs[1])

        
        #start_time = time.time()
        mem = SetHashedDict()
        for y in ints_y.keys():
            partition = 0
            for x in ints_x.keys():
                f_xy = np.array(ints_x[x]).dot(np.array(ints_y[y]))
                mem[x, y] = mem[y, x] = f_xy
                partition += f_xy

            mem.set_coo_mem((y, partition))
        #end_time = time.time()
        #print("Product sum: {}".format(end_time-start_time))

        distribution = SetHashedDict()
        for x in omega_x:
            for y in omega_y:
                distribution[x, y] = mem[x, y] / mem.get_coo_mem(y)

        self.prob_distributions['P_' + '|'.join(rvs)] = distribution
        self.xy_logits['_'.join(rvs)] = mem
        self.xy_logits['_'.join(rvs[::-1])] = mem      


    def _joint(self, rvs=['X', 'Y']):
        assert (
            isinstance(rvs, list) or
            isinstance(rvs, tuple) and
            len(rvs) == 2), "Check 'rvs' argument for two RVs"
        assert len(set(rvs).intersection(df.columns)) == 2, (
            "One or both RVs are not in df.columns")
 
        if ('_'.join(rvs) in self.xy_logits.keys()
                and self.joint_method != 'conditional'):
            ## Use partitions as multipliers. But, is it possible?
            try:
                X = self.set_rvs[rvs[0]]
                Y = self.set_rvs[rvs[1]]
                P_ygx = self.df['P_' + '|'.join(rvs)].values
            except KeyError:
                X = [
                    set(self.analyzer(x)) for x in self.df[rvs[0]].values]
                Y = [
                    set(self.analyzer(y)) for y in self.df[rvs[1]].values]
                P_ygx = self.df['P_' + '|'.join(rvs)].values
                
            mem = self.xy_logits['_'.join(rvs)]
            T_x = []
            f_ygx_times_partition = []
            
            for x, p_ygx in zip(X, P_ygx):
                partition = mem.get_coo_mem(x)
                T_x.append(partition)
                f_ygx_times_partition.append(p_ygx * partition)
                
            T_xy = sum(T_x)

            self.df['P_' + ','.join(rvs)] = np.array(f_ygx_times_partition) / T_xy

        elif self.joint_method != 'conditional':
            self._conditional(rvs=rvs)
            self._joint(rvs=rvs)
            
        else:  # P(X,Y) = P(Y|X)P(X) = f(x,y)/T_xy
            try:  
                mem = self.xy_logits['_'.join(rvs)]
                inverted = False
            except KeyError:
                try:
                    mem = self.xy_logits['_'.join(rvs[::-1])]
                    inverted = True
                except KeyError:
                    self._conditional(rvs=rvs)
                    mem = self.xy_logits['_'.join(rvs)]
                    inverted = False
                        
            try:
                X = self.set_rvs[rvs[0]]
                Y = self.set_rvs[rvs[1]]

            except KeyError:
                X = [set(self.analyzer(x)) for x in self.df[rvs[0]]]
                Y = [set(self.analyzer(y)) for y in self.df[rvs[1]]]

            T_xy = 0
            f_xy = []
            
            for x, y in zip(X, Y):
                T_xy += mem.get_coo_mem(x if (x, y) in mem else y)  # Adding partitions on 'X'
                f_xy.append(mem[(x, y)] if (x, y) in mem else mem[(y, x)])

            self.df['P_' + ','.join(rvs)] = np.array(f_xy) / T_xy        
            
       
    def fit(self, df, it_rvs=['X,Y', 'Y,Z', 'Z,X']):
        assert isinstance(df, pd.DataFrame) # Input data must a pd.DataFrame
        assert len(df.columns) >= 2 # Two or more columns are needed
        
        self.it_rvs = it_rvs
        self.df = df
        pairsRV = [rvs.split(',') for rvs in self.it_rvs]
        
        # Estimate Marginal distributions
        RVs = list(set(sum(pairsRV, [])))
        
        assert len(RVs)/len(set(df.columns).intersection(RVs)) == 1.0, ("One or"
            " more of the given columns in 'it_rvs' argument to fit() is not"
            "in the columns of the input dataframe.")
            
        _ = [self._marginal(RV) for RV in RVs]
        # Estimating conditionals and Information-Theoretic metrics
        for pair in pairsRV:
            self._conditional(pair)
            key = 'I(' + ','.join(pair) + ')'
            self.it_metrics[key] = self._mutual_info(pair)



def step_processing(df):
# Create toy Random Sets in Pandas DF.
#    df = pd.DataFrame([
#        {'X': " ".join(np.random.randint(2000, size=(1,5))[0].astype(str)),
#         'Y': " ".join(np.random.randint(2000, size=(1,5))[0].astype(str)),
#         'Z': " ".join(np.random.randint(2000, size=(1,5))[0].astype(str))
#        } for _ in range(320)
#    ])
    
    #start_time = time.time()
    df = df.dropna()[columns[1:]]
    print("Number of rows {}".format(len(df)))
    random_sets = RandomSetDistributions(df, joint_method='conditional')
    random_sets.fit(df)
    
    #end_time = time.time()
    #print("STEP time: {}".format(end_time-start_time))

    return random_sets.it_metrics
    

if __name__ == "__main__":

    start_time = time.time()
    chunk_size = 320
    input_triplets = 'data/dis_train.txt.oie'


    df_whole = pd.read_csv(
        input_triplets, sep='\t', names=columns, chunksize=chunk_size)

    results = Parallel(prefer='processes', n_jobs=-1)(
            delayed(step_processing)(df)
        for df in df_whole)
    
    end_time = time.time()
    print("Experiment time: {}".format(end_time-start_time))
    
    pd.DataFrame(results).to_csv("it_test_python.csv")

