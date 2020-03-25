from functools import partial
from jellyfish import damerau_levenshtein_distance as dlbsh
from srl_env_v01 import textEnv
import itertools
import random
import numpy as np
import pandas as pd
import re
import os
import math
import sys
import logging
import argparse
import inspect
import string
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from joblib import parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import time
import warnings

from pdb import set_trace as st

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

ANALYZER = 'char'
STRUCTCOLS = ['X', 'Y', 'Z']
COMBINATIONS= ['Y+Z', 'X+Z', 'X+Y']
BACKEND = 'mp'

#@delayed
#@wrap_non_picklable_objects


# TODO: Verify how wrapping decorators work as they seem to be faster
#@delayed
#@wrap_non_picklable_objects


def rdn_range(a, b, N=10):
    """
    This method generates tuples of random pairs for n-gram ranges.
    The parameters must meet 
    
        N < (((a - b) ** 2) - (a - b)) / 2
        
    where a < b are the interval's start and end, respectively. N is
    the desired number of pair tuples of integers.
    """
    possible = []
    n = (a - b)
    n = ((n ** 2) - n) / 2
    if n < N:
        print(
            "ERROR: N = {}, and (((a - b) ** 2) - (a - b)) / 2 = {}" \
                .format(N, n))
        return "ERROR"
    for x in range(a, b):
        for y in range(a, b):
            xy = tuple(sorted((x, y)))
            if x != y and not xy in possible:
                possible.append(xy)
    
    for tup in random.sample(possible, N):
        yield tup


class srl_env_test(object):
    """
    This class takes a csv of openIE triplets and the plain text where they
    were extracted from as inputs. The process consists in compute
    entropy-based metrics between the items of the openIE triplets, as
    well as between the items of randomly defined triplets. The results of
    the computation are deposited as a csv file specified with the --output
    argument of this script.
    ...
    
    Attributes
    ----------
    
    """

    def __init__(self, in_oie, in_txt, output_dir, sample=0, wsize=10,
                        nsteps=200, njobs=-1, analyzer='char', n_trajectories=2,
                        toanalyze = "analysis_cols.txt"):
        """
        Parameters
        ----------
        in_oie : str
            Input open IE triplets in csv.
        in_txt : str
            Input plain text where triplets were extracted.
        nsteps : int
            Number of steps to simulate (default: 50).
        njobs : int
            Number of cores to use for simulating (default: -1 = all available
            cores).
        dir : str
            Directory where the results will be placed.
        analyzer : str
            Analyzer for the tokenizer \in ('char', 'word'); default: 'char'
        wsize : int
            Window size for text environment samples or contexts (default: 10).
        n_trajectories : int
            Number of trajectories the random agent is going to simulate
        

        """

        self.input_oie      = in_oie
        self.input_plain    = in_txt
        self.output_dir     = output_dir
        self.njobs          = njobs
        self.n_steps        = nsteps
        self.n_trajectories = n_trajectories
        self.toanalyze      = toanalyze
        self.input_plain    = input_plain
        self.wsize          = wsize
        self.n_trajectories = n_trajectories
        
        self.analyzer = CountVectorizer(analyzer=analyzer, ngram_range=ngramr)\
                                .build_analyzer()
        self.gsAkdf = self._simulate_oie_actions()
        self.rdn_Akdf = self._simulate_rdn_actions()


    def rdn_partition(self, state):
        tokens = state.split()
        idxs = sorted(random.sample(list(range(1, len(tokens) - 1)), 2))
        action = np.split(tokens, idxs)
        return {c: " ".join(a) for c, a in zip(STRUCTCOLS, action)}
    

    def zlog(self, p):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                return -p[1] * np.log2(p[1] / p[0])
            except Warning as e:
                logging.warning("Warningx: %s" % e)
                return 0.0
            except ZeroDivisionError as e:
                logging.warning("Errorx: %s" % e)
                return 0.0


    def _simulate_rnd_actions(self):
        """
        Create text environment sampler to simulate random actions.
        """
        env = textEnv(input_file_name=self.input_plain, wsize=self.wsize,
                traject_length=self.n_steps, n_trajects=self.n_trajectories,
                beta_rwd=1.5, sample_size=self.sample)
        env.reset()
        S, _, done, _ = env.step()
        A = []
        logging.info("Simulating random actions from file '{}'" \
                        .format(input_plain))
        for t in range(self.n_steps):
            Ak = [self.rdn_partition(s[0]) for s in S]
            S, _, done, _ = env.step()
            A.append(Ak)
            if done:
                break

        return pd.DataFrame(sum(A, []))


    def simulate_oie_actions(self):
        with open(self.toanalyze) as f: 
            cols_results = f.readlines()
        self.toanalyze = []
        
        for s in cols_results:
            c = s.strip()
            if not c.startswith('#'):
                exec("self.toanalyze.append(" + c.strip() + ")")

        logging.info("Reading input file '{}'".format(input_oie))
        # Randomize the input gold standard
        self.gsAkdf = pd.read_csv(input_oie, delimiter='\t',
                                    keep_default_na=False,
                                    names=['score'] + STRUCTCOLS)[STRUCTCOLS] \
                                        .sample(frac=1.0)
        # Remove triplets with only punctuation
        for c in STRUCTCOLS:
            self.gsAkdf[c] = self.gsAkdf[c].apply(self._clean)

        return self.gsAkdf[
                    ~self.gsAkdf[STRUCTCOLS].isin(['', ' '])
                                           .apply(np.any, axis=1)]


    def expset(self, S, ngramr=(1, 3), sigma=1.0, bias=0.1):
        S = list(S)
        s = self.analyzer(S[0])
        hit_miss = [self.analyzer(m) for m in S[1:]]
        inner = lambda a, b: len(set(a).intersection(b))
        probability = lambda filter_trap: np.mean([
                            sigma * math.exp(
                                -sigma * inner(filter_trap, hm) + bias)
                                                  for hm in hit_miss])
        return probability(s)


    def gausset(self, S, ngramr=(1, 3), sigma=1.0, bias=1.0):
        S = list(S)
        s = self.analyzer(S[0])
        hit_miss = [self.analyzer(m) for m in S[1:]]
        metric = lambda a, b: len(set(a).union(b) - set(a).intersection(b))
        probability = lambda filter_trap: np.mean([
                                      np.sqrt(sigma/np.pi) * math.exp(
                                          -sigma * (
                                            metric(filter_trap, hm) + bias)**2)
                                                  for hm in hit_miss])

        return probability(s)


    def setmax(self, S, ngramr=(1, 3), sigma=1.0, bias=0.1):
        """This function takes a row 'S' where the fisrt item S[0] is the string
        we want to know its probability (likelihood), with respect to the 
        reamining items.
        The returned value es the needed Bayesian probability (a real number
        in [0, 1]). This is the Boltzman (softmax) distribution defined on 
        Linguistic Random Sets (setmax) taking into account S[1:] as evidence 
        normalizing the exponential and making setmax a density/pmass function.
        Observation: Notice that setmax does not distinguish instances where S[0] 
        is true and where it is not true so as to collect evidence, but rather it
        measures how much S[0] may be observed (its possibility) through all of the
        remaining S[1:]s (a Bayesian normalizing integral). To do this, each of the
        remaining S[1:] are considered an S[0] separately and its possibility is
        collected as evidence. Take care of this observation before using setmax as
        a density on sets. In additon, it may require O(n^2) extra computations.
        """
        S = list(S)
        s = self.analyzer(S[0])
        hit_miss = [self.analyzer(m) for m in S[1:]]
        # We need to specify the sign given cardinalities are always positive
        measure = lambda a, b: len(set(a).intersection(b))
        likelihood = lambda filter_trap: np.mean([sigma * math.exp(
                                     sigma * bias) * math.exp(
                                       -sigma * measure(filter_trap, hm) + bias)
                                                            for hm in hit_miss])
        rlh = likelihood(s)
        # Evicence, O(n^2) !!
        evidence = [likelihood(s) for s in hit_miss]
        evidence.append(rlh)
        return rlh/sum(evidence)
    

    def compute_set_probability(self, Akdf, prod_cols, hit_miss_samples=50, sigma=5.0, 
                            ngramr=(1, 3), density='setmax', bias=1.0):
        try:
            assert len(Akdf.index) >= hit_miss_samples, (
                                                "The number of hit-and-miss"
                                                " samples must be less or equal"
                                                " than the number of samples")
        except AssertionError:
            return None
    
        if density == 'gausset':
            capacity = partial(gausset, ngramr=ngramr, sigma=sigma, bias=bias)
        if density == 'setmax':
            capacity = partial(setmax, ngramr=ngramr, sigma=sigma)
        if density == 'expset':
            capacity = partial(expset, ngramr=ngramr, sigma=sigma, bias=bias)
        
        joints = []
        for d in prod_cols:
            if '+' in d[0]:
                joints.append(tuple(d[0].split('+')))
            elif '+' in d[1]:
                joints.append(tuple(d[1].split('+')))
            else:
                continue

        for a, b in joints:
            Akdf['+'.join((a, b))] = Akdf[[a, b]].apply(lambda x: ' '.join(x),
                                                          axis=1)
        for a, b in prod_cols:  # This lists already contains joints
            measures = []
            rdns = ["b_" + str(i) for i in range(hit_miss_samples)]
            dupst = np.array([Akdf[b].values] * hit_miss_samples)
            _ = np.apply_along_axis(np.random.shuffle, 1, dupst)
            rdns_df = pd.DataFrame(dupst.T, columns=rdns)
            trap_df = Akdf[[a, b]]
            if a == b:
                trap_df.set_axis([a, b + '_'], axis=1, inplace=True)
            
            to_operate = pd.concat([trap_df.reset_index(drop=True),
                                    rdns_df.reset_index(drop=True)],
                                    axis=1, sort=False)[[a] + rdns]
            joincol = "$\mathcal{{N}}\{{h(" + ', '.join((a, b)) + "), \sigma\}}$"
        
            Akdf[joincol] = to_operate.apply(capacity, axis=1).tolist()
        
        return Akdf.dropna()                


    def compute_mutuals(self, df, cols):
        scs = ''.join(STRUCTCOLS)
        patt = r"\{{h\(([{0}]\+?[{0}]?), ([{0}]\+?[{0}]?)\)" \
                        .format(scs)
        pairs = sum([re.findall(patt, c) for c in cols], [])
        selfs = ["$\mathcal{{N}}\{{h(" + ', '.join(p) + "), \sigma\}}$"
                        for p in pairs if p[0] == p[1]]
        joins = [("$\mathcal{{N}}\{{h(" + ', '.join(p) + "), \sigma\}}$",
                  "$\mathcal{{N}}\{{h(" + ', '.join([p[1], p[1]]) + "), \sigma\}}$",
                  "$\mathcal{{N}}\{{h(" + ', '.join([p[0], p[0]]) + "), \sigma\}}$")
                        for p in pairs if p[0] != p[1]]
        entropy = lambda x: -x * np.log2(x) if x > 0.0 else 0.0
        centropy = lambda x: self.zlog(x)
        for s in selfs:
            try:
                icol = "$H[h(" + ', '.join(re.findall(patt, s)[0]) + ")]$"
                df[icol] = df[s].apply(entropy)
            except:
                st()
        for j in joins:
            rsets = re.findall(patt, j[0])[0]
            icol = "$I[h(" + ', '.join([rsets[1], rsets[0]]) + ")]$"
            try:
                df[icol] = df[list(j)].apply(lambda p: entropy(p[1]) - centropy(p),
                                             axis=1)
            except:
                st()

        return df[[c for c in df.columns
                    if True in ("$I[" in c, "$H[h(" in c)]].sum().to_dict()


    def compute_mi_steps(self, Akdf, out_csv, sample_size, sigma=5.0, prod_cols=None,
                            bias=1.0, density='setmax', n_hit_miss=50, ngramr=(1, 3)):
    """ This method calls 'compute_set_probability()' 
    """
        probcs = []
        if prod_cols is None:
            prod_cols = []
            for a, b in itertools.product(*[STRUCTCOLS + COMBINATIONS] * 2):
                if not ((a, b) in prod_cols or (b, a) in prod_cols):
                    probcs.append("$\mathcal{{N}}\{{h({0}, {1}), \sigma\}}$" \
                                .format(a, b))
                    prod_cols.append((a, b))
        else:
            for a, b in prod_cols:
                probcs.append("$\mathcal{{N}}\{{h({0}, {1}), \sigma\}}$" \
                                .format(a, b))
                        
        A_tau = [Akdf[i:i + sample_size]
                for i in range(0, self.n_steps * sample_size, sample_size)]
    
        logging.info(f"Computing probabilities of random sets for {N_STEPS} steps.")
        with parallel_backend('multiprocessing' if BACKEND == 'mp' else 'loky'):
            t = time.time()
            P_Aks = Parallel(n_jobs=self.njobs)(
                        delayed(self.compute_set_probability)(
                            A_k, prod_cols=prod_cols, hit_miss_samples=n_hit_miss,
                            density=density, bias=bias, sigma=sigma, ngramr=ngramr)
                                                                    for A_k in A_tau)
            logging.info("Estimated set probabilities in {}s..." \
                        .format(time.time() - t))
    
        with parallel_backend('multiprocessing' if BACKEND == 'mp' else 'loky'):
            t = time.time()
            info_steps = Parallel(n_jobs=self.njobs)(
                            delayed(self.compute_mutuals)(df, probcs)
                                for df in P_Aks if not (df is None or df.empty))
            logging.info("Estimated MIs in {}s..." \
                                .format(time.time() - t))
                                
        pd.DataFrame(info_steps).to_csv(out_csv)


    def _clean(self, x):
        translator = str.maketrans('', '', string.punctuation)
        try:
            return x.translate(translator)
        except AttributeError:
            return "__NULL__"


    def _make_output_name(self, namespace, nonh=["in_oie", "in_txt", "output",
                                                    "njobs", "self.output_dir"]):
        names = [a + "-" + "t".join(map(str, a[1]))
                    if isinstance(a[1], tuple)
                    else "-".join(map(str, a))
                        for a in namespace.items()
                            if not a[0] in nonh]
                        
        return "_".join(names) + ".csv"
            

    def fit(self, wsize, bias, hitmiss=0.0, bw=5.0, sample=100, density='expset',
                    ngrams=(1, 3), output=None):
        """
        Parameters
        ----------
                        
        ngrams : tuple([int, int])
            N-gram range to form elementary text strings to form sets 
            default: (1, 3) set as '1 3' (two space-separated integers).
        output : str
            Output results in csv. (default: None --> 'rdn_<params-values>.csv' and
            'oie_<params-values>.csv').
        sample : int
            Sample size for text environment steps (default: 100).
        density : str
            Density function/kernel estimator. ('expset', 'gausset', 'setmax';
            default: 'gausset').
        bw : float
            Bandwidth/strinctness for the kernel estimator. (default: 5.0)
        hitmiss : float
            Number of samples to build the hit-and-missing topology
            (in [0.0, 1.0]; default: 0.25: 25%% of sample size).
        bias : float
            Bias parameter for linear separator densities (default: 1.0).
        """

        self.properties = {k: v for k, v in locals().items()
                          if k in inspect.getfullargspec(self.fit).args + ['output']}
        if output is None:
            self.out_name = self._make_output_name(self.properties)
            self.donot_make_oie, self.donot_make_rdn = (
                            os.path.isfile(self.output_dir + "/oie_" + self.out_name),
                            os.path.isfile(self.output_dir + "/rdn_" + self.out_name))
        else:
            self.out_name = output
            self.donot_make_oie, self.donot_make_rdn = (
                            os.path.isfile(self.output_dir + "/oie_" + self.out_name),
                            os.path.isfile(self.output_dir + "/rdn_" + self.out_name))

        logging.info("Processing input parameters:\n{}\n" \
                        .format(self.out_name \
                                    .replace('-', ' : ')[5:] \
                                    .replace('_', '\n')))
        t_start = time.time()
        n_hit_miss = int(sample * hitmiss)
    
        if donot_make_oie:
            logging.info("MI for OpenIE actions already exists (SKIPPED)...")
        else:
            logging.info("Computing MI for OpenIE actions...")

            self.compute_mi_steps(self.gsAkdf, prod_cols=self.toanalyze,
                    out_csv=self.output_dir + "/oie_" + out_name,
                    sigma=bw, density=density,
                    ngramr=ngramr, n_hit_miss=n_hit_miss, bias=bias)

        if donot_make_rdn:
            logging.info("MI for random actions already exists (SKIPPED)...")
        else:
            logging.info("Computing MI for random actions...")
            self.compute_mi_steps(rdn_Akdf, prod_cols=TOANALYZE, density=args.density,
                    out_csv="rdn_" + rdn_out_name,
                    sigma=args.bw, ngramr=ngramr, n_hit_miss=n_hit_miss)
                    
        logging.info("Results saved to: \n{}\n{}" \
                                            .format("oie_" + oie_out_name,
                                                    "rdn_" +  rdn_out_name))
        logging.info("Terminated {} in {}s..." \
                        .format(self.output_dir + "/[oie|rdn]_" + out_name,
                                time.time() - t_start))
