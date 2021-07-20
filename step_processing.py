from sklearn.feature_extraction.text import CountVectorizer
from joblib import Parallel, delayed
from functools import partial
from scipy.stats import entropy
import numpy as np
import pandas as pd
import math
import time
from functools import partial
from pdb import set_trace as st

# ------------------------------------------------------------------------------
max_probs = {}
min_probs = {}
max_itms = {}
min_itms = {}
# ------------------------------------------------------------------------------

class SetHashedDict:
    
    def __init__(self):
        self.x_keys = []
        self.y_keys = []
        self._values = {}
        self.coo_keys = []
        self.coo_mem = {}
        

    def __getitem__(self, ks):
        if isinstance(ks, list) or isinstance(ks, tuple):
            ka, kb = ks
            ix = self.x_keys.index(ka)
            iy = self.y_keys.index(kb)

            return self._values[(ix, iy)]
        else:
            ix = self.x_keys.index(ks)
            return self._values[ix]
        

    def __setitem__(self, ks, value):
        if isinstance(ks, list) or isinstance(ks, tuple):
            ka, kb = ks
            if not ka in self.x_keys:
                self.x_keys.append(ka)

            if not kb in self.y_keys:
                self.y_keys.append(kb)
        
            self._values[(
                self.x_keys.index(ka),
                self.y_keys.index(kb)
            )] = value
        else:
            if not ks in self.x_keys:
                self.x_keys.append(ks)
            
            self._values[self.x_keys.index(ks)] = value
        

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
            for it in self._values.items():
                yield (
                    self.x_keys[it[0][0]],
                    self.y_keys[it[0][1]], it[1])
        else:
            for it in self._values.items():
                yield (self.x_keys[it[0]], it[1])


    def values(self):
        return self._values.values()


    def set_coo_mem(self, ks):
        assert isinstance(ks, list) or isinstance(ks, tuple)
        k, value = ks
        if not k in self.coo_keys:
            self.coo_keys.append(k)
        
        self.coo_mem[self.coo_keys.index(k)] = value
        
        
    def get_coo_mem(self, ks):
        assert isinstance(ks, set)
        idx = self.coo_keys.index(ks)

        return self.coo_mem[idx]


class RandomSetDistributions(object):

    def __init__(self,
            gamma=None, joint_method='conditional', kernel=None, n_jobs=-1,
            backend='processes', ngram_range=(1, 4), analyzer='char_wb'):
        self.gamma = gamma
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.projs = {}
        self.n_jobs = n_jobs
        self.backend = backend
        self.kernel = kernel
        self.set_rvs = {}
        self.distributions = {}
        self.it_metrics = {}
        self.Omega = {}
        self.check_np = lambda f : type(f).__module__ == np.__name__
        self.tokenizer = self.build_tokenizer()


    def build_tokenizer(self):
        tokenizer = CountVectorizer(
            analyzer=self.analyzer, ngram_range=self.ngram_range)
        return tokenizer.build_analyzer()


    def _build_vocab(self, rv):
        omega = []
        for x in self.set_rvs[rv]:
            if x in omega:
                continue
            else:
                omega.append(x)
                
        return omega


    def _build_set_outcomes(self, rv):
        sets = []
        for x in self.df[rv]:
            sets.append(set(self.tokenizer(x)))

        return sets


    def _kernel(self, x):        
        try:
            if self.kernel is None:
                return np.array(x)
            elif self.kernel == 'gausset':
                return (1/np.sqrt(2 * math.pi)) * np.exp(-self.gamma * np.array(x) ** 2)
            elif self.kernel == 'expset':
                return np.exp(self.gamma * np.array(x))
            elif (isinstance(self.kernel, collections.abc.Callable)
                    and self.check_np(self.kernel)):
                return self.kernel(np.array(x)) 
            else:
                print("WARNING: Bad kernel function specified."
                        " Only intersect returned")
                return np.array(x)
        except OverflowError:
            print("OverflowError in kernel"
               " computations: f({}; {}) undetermined".format(x, self.gamma))
            return None


    def _projector(self, x, y):
        if self.kernel == 'gausset':
            # It would be great to include normalized distances
            # \cite{conci2018distance,app11041910}
            return len((x - y).union(y - x))
        elif self.kernel == 'expset':
            # # It would be great to include similarity metrics 
            return len(x.intersection(y))
        else:
            return len(x.intersection(y))


    def _build_projections(self, rv):
        rv_projections = SetHashedDict()
        for x in self.Omega[rv]:
            projection = []
            for x_ in self.set_rvs[rv]:
                projection.append(self._projector(x, x_))

            rv_projections[x] = projection
            
        return rv_projections


    def _marginal(self, rv='X'):
        assert isinstance(rv, str)

        try:
            self.Omega[rv] = self._build_vocab(rv)
        except:
            self.set_rvs[rv] = self._build_set_outcomes(rv)
            self.Omega[rv] = self._build_vocab(rv)
        
        self.projs[rv] = self._build_projections(rv)

        mem = SetHashedDict()
        partition = 0
        for x, proj in self.projs[rv].items():
            f_X = self._kernel(proj).sum()
            mem[x] = f_X
            partition += f_X

        distribution = SetHashedDict()
        for x in self.Omega[rv]:
            distribution[x] = mem[x] / partition
            
        self.distributions['P_' + rv] = distribution


    def _conditional(self, rvs=['X', 'Y']):
        assert (isinstance(rvs, list) or
                isinstance(rvs, tuple) and
                len(rvs) == 2), "Check 'rvs' argument for two RVs"
        try:
            omega_x = self.Omega[rvs[0]]
        except KeyError:
            self.Omega[rvs[0]] = self._build_vocab(rvs[0])
            omega_x = self.Omega[rvs[0]]
        try:
            omega_y = self.Omega[rvs[1]]
        except KeyError:
            self.Omega[rvs[1]] = self._build_vocab(rvs[1])
            omega_y = self.Omega[rvs[1]]
        try:
            projs_x = self.projs[rvs[0]]
        except KeyError:
            projs_x = self._build_projections(rvs[0])
        try:
            projs_y = self.projs[rvs[1]]
        except KeyError:
            projs_y = self._build_projections(rvs[1])
        # (1996) Estimating and Visualizing Conditional Densities [Hyndman - Bashtannyk - Grunwald]
        mem = SetHashedDict()
        for y, proj_y in projs_y.items():
            partition = 0
            for x, proj_x in projs_x.items():
                f_xy = self._kernel(proj_x).dot(self._kernel(proj_y))
                mem[x, y] = f_xy
                partition += f_xy

            mem.set_coo_mem((y, partition))

        distribution = SetHashedDict()
        for x in omega_x:
            for y in omega_y:
                distribution[x, y] = mem[x, y] / mem.get_coo_mem(y)

        self.distributions['P_' + '|'.join(rvs)] = distribution


    def _entropy(self, rv):
        P_X = self.distributions['P_' + rv]
        H_X = entropy(list(P_X.values()), base=2)
        self.it_metrics['H(' + rv + ')'] = H_X
        
        return H_X


    def _cond_entropy(self, rvs=['X', 'Y']):
        P_XgY = self.distributions['P_' + '|'.join(rvs)]
        P_Y = self.distributions['P_' + rvs[1]]
        omega_x = self.Omega[rvs[0]]
        omega_y = self.Omega[rvs[1]]
        
        H_Xgy = [
            P_Y[y] * entropy(
                [P_XgY[x, y] for x in omega_x], base=2)
                    for y in omega_y]
        H_XgY = sum(H_Xgy)
        self.it_metrics['H(' + '|'.join(rvs) + ')'] = H_XgY
        
        return H_XgY

        
    def _mutual_info(self, rvs=['X', 'Y']):
        try:
            H_X = self.it_metrics['H(' + rvs[0] + ')']
        except KeyError:
            H_X = self._entropy(rvs[0])
        try:
            self.it_metrics['H(' + '|'.join(rvs) + ')']
        except KeyError:
            H_XgY = self._cond_entropy(rvs)

        I_XY = H_X - H_XgY
        self.it_metrics['I(' + ';'.join(rvs) + ')'] = I_XY
        
        return I_XY

       
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
        
        for RV in RVs:
            self._marginal(RV)
        # Estimating pair-wise Information-Theoretic metrics

        for pair in pairsRV:
            self._conditional(pair)
            self._mutual_info(pair)
            # Change to this method if no problem when returnning the metrics:
            #key = 'I(' + ','.join(pair) + ')'
            #self.it_metrics[key] = self._mutual_info(pair)

def save_extrema(set_statistics):

    global max_probs
    global min_probs
    global max_itms
    global min_itms
    if max_probs == {}:
        max_probs = {k: [0, ''] for k in set_statistics.distributions.keys()}
    if min_probs == {}:
        min_probs = {k: [1.5, ''] for k in set_statistics.distributions.keys()}
    if max_itms == {}:    
        max_itms = {k: [0, ''] for k in set_statistics.it_metrics.keys()}
    if min_itms == {}:
        min_itms = {k: [1000, ''] for k in set_statistics.it_metrics.keys()}

    for k, vs in set_statistics.distributions.items():
        mean_prob = np.mean(list(vs.values()))
        if mean_prob > max_probs[k][0]: 
            max_probs[k] = (mean_prob, set_statistics)
        if mean_prob < min_probs[k][0]:
            min_probs[k] = (mean_prob, set_statistics)

    for k, vs in set_statistics.it_metrics.items():
        if vs > max_itms[k][0]:
            max_itms[k] = (vs, set_statistics)
        if vs < min_itms[k][0]:
            min_itms[k] = (vs, set_statistics)


def step_processing(
    df, it_rvs=['Y,X', 'Z,Y', 'X,Z'], kernel='gausset', gamma=1.0/50.0):
    
    start_time = time.time()

    df = df.dropna()
    random_sets = RandomSetDistributions(kernel=kernel, gamma=gamma)
    random_sets.fit(df, it_rvs=it_rvs)
    # --------------------------------------------------------------------------
    save_extrema(random_sets)
    # --------------------------------------------------------------------------
    end_time = time.time()
    print("STEP time: {}".format(end_time-start_time))

    return random_sets.it_metrics
    

if __name__ == "__main__":

    start_time = time.time()
    chunk_size = 220
    input_triplets = 'data/dis_train.txt.oie'
    it_rvs = ['Y,X', 'Z,Y', 'X,Z']  # Order verified from De Marcken (1999)
    kernel = 'gausset'
    #kernel = 'expset'
    gamma = 1.0/50.0
    output_it = "results/it_{}_kernel-{}_gamma-{}_sample-{}.csv".format(
        'train' if '_train.' in input_triplets else 'test',
        kernel,
        gamma,
        chunk_size)
    n_jobs = 1
    columns = [1, 2, 3]
    names = ['X', 'Y', 'Z']

    # Generate chunks from input csv triplets
    df_generator = pd.read_csv(
        input_triplets,
        sep='\t',
        names=names,
        chunksize=chunk_size,
        usecols=columns)

    step = partial(step_processing, it_rvs=it_rvs, kernel=kernel, gamma=gamma)
    results = Parallel(
        n_jobs=n_jobs, require='sharedmem'
        #prefer='processes', n_jobs=n_jobs
        )(
            delayed(step)(df)
        for df in df_generator)
    st()    
    end_time = time.time()
    print("Experiment time: {}".format(end_time-start_time))
    
    pd.DataFrame(results).to_csv(output_it)
