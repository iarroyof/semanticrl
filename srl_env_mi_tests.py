from functools import partial
from jellyfish import damerau_levenshtein_distance as dlbsh
from srl_env_v01 import textEnv
import itertools
import random
import numpy as np
import pandas as pd
import re
import math
import logging
import argparse
import string
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from joblib import parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import time

from pdb import set_trace as st

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

ANALYZER = 'char'
STRUCTCOLS = ['X', 'Y', 'Z']
COMBINATIONS= ['Y+Z', 'X+Z', 'X+Y']
BACKEND = 'mp'


def dot_distance(A, B, binary=True, euclid=False, ngramr=(1, 3)):
    """ This method computes either Euclidean or Hamming distance between
    two strings 'A' and 'B'. The input strings are encoded either as 'binary'
    (True) or as BoW (False) vectors. The 'euclid' (True) parameter is thought
    for BoW vectors and Euclidean distance. The case of BoW vectors operated
    by Hamming distance (binary=False, euclid=True) is not defined, but some
    value is returned.
    The 'ngramr' parameter defined n-grams range into which the strings are
    segmented. The 'ANALYZER' constant can be modified from 'char' to 'wb_char'
    for sklearn >= 21.0 versions. Also 'word' is allowed.
    """
    try:
        assert isinstance(A, str) and isinstance(B, str)
    except AssertionError:
        if np.nan in [A, B]:
            return 10000.0

    vectorizer = CountVectorizer(binary=binary, analyzer=ANALYZER,
                                 ngram_range=ngramr)

    X = vectorizer.fit_transform([A, B])
    if euclid:
        return (X[0] - X[1]).dot((X[0] - X[1]).T).sum()
    else:
        return (X[0].toarray() ^ X[1].toarray()).sum()


def set_valued_gaussian(S, M, sigma=5.0, metric='hmm', ngramr=(1, 3)):
    """ The metric can be Levenshtein ('lev') or Hamming ('hmm') or Euclidean
    ('euc', this encodes strings as simple BoW vectors).
    Other parameters are inherited from the dot_distance() method.
    """
    if metric == 'lev':
        assert isinstance(S, str) and isinstance(M, str)
        distance = dlbsh(S, M)
    elif metric == 'hmm':
        A = set(analyzer(S))
        B = set(analyzer(M))
        # |Symmetric difference| == sum(xor)
        distance = len(A.union(B) - A.intersection(B))
    elif metric == 'euc':
        distance = dot_distance(S, M, binary=False,
                                            euclid=True, ngramr=ngramr)

    if distance <= 1.0:
        return 1.0 / (np.sqrt(2 * np.pi * sigma ** 2))
    else:
        return (1.0 / (np.sqrt(2 * np.pi * sigma ** 2))) * math.exp(
                                             -distance ** 2 / (2 * sigma ** 2))


def gausset(S, ngramr=(1, 3), sigma=5.0):
    S = list(S)
    s = analyzer(S[0])
    hit_miss = [analyzer(m) for m in S[1:]]
    measure = lambda a, b: len(set(a).union(b) - set(a).intersection(b))
    likelihood = lambda filter_trap: sum([math.exp(
                                            -measure(filter_trap, hm)**2/
                                                               2*sigma**2)
                                                          for hm in hit_miss])
    rlh = likelihood(s)
    evidence = [likelihood(s) for s in hit_miss]
    evidence.append(rlh)
    return rlh/sum(evidence)
                                                                

def setmax(S, bias=None, ngramr=(1, 3), sigma=1.0):
    """This function takes a row 'S' where the fisrt item S[0] is the string
    we want to know its probability, with respect to the reamining items.
    The returned value es the needed Bayesian probability (a real number
    in [0, 1]). This is the Boltzman (softmax) distribution defined on 
    Linguistic Random Sets (setmax).
    """
    S = list(S)
    s = analyzer(S[0])
    hit_miss = [analyzer(m) for m in S[1:]]
    # We need to specify the sign given cardinalities are always positive
    measure = lambda a, b: len(set(a).intersection(b))
                  #if comparer == 'intersect' else len(set(a).union(b)
                  #                                - set(a).intersection(b)))
    likelihood = lambda filter_trap: sum([math.exp(
                                            -sigma * measure(filter_trap, hm))
                                                            for hm in hit_miss])
    # Likelihood
    rlh = likelihood(s)
    # Evicence
    evidence = [likelihood(s) for s in hit_miss]
    evidence.append(rlh)
    return rlh/sum(evidence)


#@delayed
#@wrap_non_picklable_objects
def compute_set_probability(Akdf, prod_cols, hit_miss_samples=50, sigma=5.0,
                            metric='hmm', ngramr=(1, 3), density='setmax'):
    try:
        assert len(Akdf.index) >= hit_miss_samples, (
                                                "The number of hit-and-miss"
                                                " samples must be less or equal"
                                                " than the number of samples")
    except AssertionError:
        return None
    
    if density == 'gausset':
        capacity = partial(gausset, ngramr=ngramr, sigma=sigma)
    if density == 'setmax':
        capacity = partial(setmax, ngramr=ngramr, sigma=sigma)
    
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
        #for _ in range(hit_miss_samples):
        #    B = Akdf[b].sample(frac=1)
        #    measures.append(
        #        np.vectorize(set_valued_gaussian,
        #                         excluded=set(['ngramr']))(
        #                Akdf[a].str.lower(), B.str.lower(), sigma=sigma,
        #                metric=metric, ngramr=ngramr
        #                )
        #        )
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
        
        #Akdf[joincol] = np.vstack(measures).mean(axis=0)
    return Akdf.dropna()


def rdn_partition(state):
    tokens = state.split()
    idxs = sorted(random.sample(list(range(1, len(tokens) - 1)), 2))
    action = np.split(tokens, idxs)
    return {c: " ".join(a) for c, a in zip(STRUCTCOLS, action)} 

# TODO: Verify how wrapping decorators work as they seem to be faster
#@delayed
#@wrap_non_picklable_objects
def compute_mutuals(df, cols):
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
    centropy = lambda x: -x[1] * np.log2(x[1] / x[0]) \
                            if x[0] > 0 and x[1] > 0 else 0.0
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


def compute_mi_steps(Akdf, out_csv, metric='hmm', sigma=5.0, prod_cols=None,
                              density='setmax', n_hit_miss=50, ngramr=(1, 3)):
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
                        
    A_tau = [Akdf[i:i + SAMPLE_SIZE]
                for i in range(0, N_STEPS * SAMPLE_SIZE, SAMPLE_SIZE)]
    
    logging.info(f"Computing probabilities of random sets for {N_STEPS} steps.")
    with parallel_backend('multiprocessing' if BACKEND == 'mp' else 'loky'):
        t = time.time()
        P_Aks = Parallel(n_jobs=NJOBS)(
                    delayed(compute_set_probability)(
                        A_k, prod_cols=prod_cols, hit_miss_samples=n_hit_miss,
                        density=density, metric=metric, sigma=sigma, ngramr=ngramr)
                                                            for A_k in A_tau)
        logging.info("Estimated set probabilities in {}s..." \
                        .format(time.time() - t))
    
    with parallel_backend('multiprocessing' if BACKEND == 'mp' else 'loky'):
        t = time.time()
        info_steps = Parallel(n_jobs=NJOBS)(
                    delayed(compute_mutuals)(df, probcs)
                             for df in P_Aks if not (df is None or df.empty))
        logging.info("Estimated MIs in {}s..." \
                                .format(time.time() - t))
                                
    pd.DataFrame(info_steps).to_csv(
                out_csv + "_Dk-{}_rho-{}_tau-{}_ng-{}.csv"
                         .format(SAMPLE_SIZE, n_hit_miss, N_STEPS, ngramr)
                         .replace(' ', '').replace(',', 'to'))


def clean(x):
    translator = str.maketrans('', '', string.punctuation)
    try:
        return x.translate(translator)
    except AttributeError:
        return "__NULL__"


parser = argparse.ArgumentParser()
parser.add_argument("--ngrams", help=("N-gram range for sklearn ngramer"
                                      " (default: (1, 3))"),
                    type=int, nargs='+')
parser.add_argument("--wsize", help=("Window size for text environment states"
                                     " (default: 10)"),
                    type=int, default=10)
parser.add_argument("--metric", help=("Metric of the measure space (default:"
                                      " 'hmm' = Hamming, other options: 'euc' ="
                                      " Euclidean, 'lev' = Levenshtein)"),
                    type=str, default='hmm')
parser.add_argument("--in_oie", help="Input open IE triplets in csv")
parser.add_argument("--in_txt", help=("Input plain text where triplets were"
                                      " extracted"))
parser.add_argument("--sample", help=("Sample size for text environment steps"
                                      " (default: 100)"),
                    type=int, default=100)
parser.add_argument("--nsteps", help=("Number of steps to simulate"
                                      " (default: 50)"),
                    type=int, default=50)
parser.add_argument("--density", help=("Density function/kernel estimator."
                                  " ('gausset', 'setmax'; default: 'gausset')"),
                                  default='gausset')
parser.add_argument("--bw", help=("Bandwidth/strinctness for the kernel estimator."
                                      " (default: 5.0)"),
                    type=float, default=5.0)
parser.add_argument("--hitmiss", help=("Number of samples to build the"
                                      " hit-and-missing topology (default: 50)"),
                    type=int, default=50)
parser.add_argument("--njobs", help=("Number of cores to use for simulating"
                                     " (default: -1 = all available cores)"),
                    type=int, default=-1)
args = parser.parse_args()

input_oie = args.in_oie
#fitting_sim_oie = "data/sim_train.txt.oie"
#develop_sim_oie = "data/sim_test.txt.oie"
#fitting_unr_oie = "data/dis_train.txt.oie"
#develop_unr_oie = "data/dis_test.txt.oie"
input_plain = args.in_txt
#fitting_sim = "data/sim_train_.txt"
#develop_sim = "data/sim_test_.txt"
#fitting_unr = "data/dis_train_.txt"
#develop_unr = "data/dis_test_.txt"
NJOBS = args.njobs
SAMPLE_SIZE = args.sample  # eg. 100
N_STEPS = args.nsteps  # 120  #e.g.: n_input_oie/SAMPLE_SIZE=12167/100=121.67
ngramr = tuple(args.ngrams)  # (1, 3)
t_size = args.wsize  # 10
N_TRAJECTORIES = 2
analyzer = CountVectorizer(analyzer=ANALYZER, ngram_range=ngramr)\
                                .build_analyzer()

if args.hitmiss == 0:
    n_hit_miss = int(SAMPLE_SIZE * 0.25)  # 25% of the sample size
else:
    n_hit_miss = args.hitmiss
    
TOANALYZE = "analysis_cols.txt"
with open(TOANALYZE) as f: 
    cols_results = f.readlines()
TOANALYZE = []
for s in cols_results:
    c = s.strip()
    if not c.startswith('#'):
        exec("TOANALYZE.append(" + c.strip() + ")")

t_start = time.time()

logging.info("Reading input file '{}'".format(input_oie))
# Randomize the input gold standard
gsAkdf = pd.read_csv(input_oie, delimiter='\t', keep_default_na=False,
                     names=['score'] + STRUCTCOLS)[STRUCTCOLS].sample(frac=1.0)
# Remove triplets with only punctuation
for c in STRUCTCOLS:
    gsAkdf[c] = gsAkdf[c].apply(clean)

gsAkdf = gsAkdf[
            ~gsAkdf[STRUCTCOLS].isin(['', ' '])
                               .apply(np.any, axis=1)            
         ]
# Take N_STEPS and compute their marginal and joint informations
logging.info("Computing MI for OpenIE actions...")

compute_mi_steps(gsAkdf, prod_cols=TOANALYZE, out_csv=input_oie.split('.')[0],
                 metric=args.metric, sigma=args.bw, density=args.density,
                    ngramr=ngramr, n_hit_miss=n_hit_miss)

# Create text environment sampler to simulate random actions
env = textEnv(input_file_name=input_plain, wsize=t_size,
                traject_length=N_STEPS, n_trajects=N_TRAJECTORIES,
                beta_rwd=1.5, sample_size=SAMPLE_SIZE)
env.reset()
S, _, done, _ = env.step()
A = []
logging.info("Simulating random actions from file '{}'".format(input_plain))
# The srl_env_class removes punctuation and empty strings before returning
# states.
for t in range(N_STEPS):
    Ak = [rdn_partition(s[0]) for s in S]
    S, _, done, _ = env.step()
    A.append(Ak)
    if done:
        break

rdn_Akdf = pd.DataFrame(sum(A, []))

logging.info("Computing MI for random actions...")
compute_mi_steps(rdn_Akdf, prod_cols=TOANALYZE, density=args.density,
                    out_csv=input_plain.split('.')[0],
                    metric=args.metric, sigma=args.bw,
                    ngramr=ngramr, n_hit_miss=n_hit_miss)
logging.info("Terminated in {}s...".format(time.time() - t_start))
