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
import warnings

from pdb import set_trace as st

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

ANALYZER = 'char'
STRUCTCOLS = ['X', 'Y', 'Z']
COMBINATIONS= ['Y+Z', 'X+Z', 'X+Y']
BACKEND = 'mp'


def make_output_name(namespace, nonh=["in_oie", "in_txt", "output", "njobs"]):
    names = [a[0] + "-" + "t".join(map(str, a[1]))
                if isinstance(a[1], list)
                else "-".join(map(str,a))
                    for a in args._get_kwargs()
                        if not a[0] in nonh]
            
    return "_".join(names) + ".csv"


def expset(S, ngramr=(1, 3), sigma=1.0, bias=0.1):
    S = list(S)
    s = analyzer(S[0])
    hit_miss = [analyzer(m) for m in S[1:]]
    inner = lambda a, b: len(set(a).intersection(b))
    probability = lambda filter_trap: np.mean([
                            sigma * math.exp(
                                -sigma * inner(filter_trap, hm) + bias)
                                                  for hm in hit_miss])
    return probability(s)


def gausset(S, ngramr=(1, 3), sigma=1.0, bias=1.0):
    S = list(S)
    s = analyzer(S[0])
    hit_miss = [analyzer(m) for m in S[1:]]
    metric = lambda a, b: len(set(a).union(b) - set(a).intersection(b))
    probability = lambda filter_trap: np.mean([
                                      np.sqrt(sigma/np.pi) * math.exp(
                                          -sigma * (
                                            metric(filter_trap, hm) + bias)**2)
                                                  for hm in hit_miss])

    return probability(s)


def setmax(S, ngramr=(1, 3), sigma=1.0, bias=0.1):
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
    s = analyzer(S[0])
    hit_miss = [analyzer(m) for m in S[1:]]
    # We need to specify the sign given cardinalities are always positive
    measure = lambda a, b: len(set(a).intersection(b))
    likelihood = lambda filter_trap: np.mean([sigma * math.exp(
                                     sigma * bias) * math.exp(
                                       -sigma * measure(filter_trap, hm) + bias)
                                                            for hm in hit_miss])
    # Likelihood
    rlh = likelihood(s)
    # Evicence, O(n^2) !!
    evidence = [likelihood(s) for s in hit_miss]
    evidence.append(rlh)
    return rlh/sum(evidence)


#@delayed
#@wrap_non_picklable_objects
def compute_set_probability(Akdf, prod_cols, hit_miss_samples=50, sigma=5.0, #metric='hmm',
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


def rdn_partition(state):
    tokens = state.split()
    idxs = sorted(random.sample(list(range(1, len(tokens) - 1)), 2))
    action = np.split(tokens, idxs)
    return {c: " ".join(a) for c, a in zip(STRUCTCOLS, action)} 

# TODO: Verify how wrapping decorators work as they seem to be faster
#@delayed
#@wrap_non_picklable_objects
def compute_mutuals(df, cols):
    def zlog(p):
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
    centropy = lambda x: zlog(x)
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


def compute_mi_steps(Akdf, out_csv, sigma=5.0, prod_cols=None, bias=1.0,
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
                        density=density, bias=bias, sigma=sigma, ngramr=ngramr)
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
                                
    pd.DataFrame(info_steps).to_csv(out_csv)


def clean(x):
    translator = str.maketrans('', '', string.punctuation)
    try:
        return x.translate(translator)
    except AttributeError:
        return "__NULL__"


parser = argparse.ArgumentParser(description=("This script takes a csv of openIE"
                                " triplets and the plain text where they were"
                                " extracted as inputs. The process consists in"
                                " compute entropy-based metrics between the"
                                " items of the openIE triplets, as well as between"
                                " the items of randomly defined triplets. The"
                                " results of the computation are deposited as a"
                                " csv file specified with the --output argument" 
                                " of this script."))
parser.add_argument("--ngrams", help=("N-gram range to form elementary text"
                                      " strings to form sets (default: (1, 3))"),
                    type=int, nargs='+', default=[1, 3])
parser.add_argument("--wsize", help=("Window size for text environment samples"
                                     " or contexts (default: 10)"),
                    type=int, default=10)
parser.add_argument("--in_oie", help="Input open IE triplets in csv")
parser.add_argument("--in_txt", help=("Input plain text where triplets were"
                                      " extracted"))
parser.add_argument("--output", help=("Output results in csv. (default:"
                    " 'rdn_<params-values>.csv' and"
                    " 'oie_<params-values>.csv')."), default=None)
parser.add_argument("--sample", help=("Sample size for text environment steps"
                    " (default: 100)"),
                    type=int, default=100)
parser.add_argument("--nsteps", help=("Number of steps to simulate"
                    " (default: 50)"), type=int, default=50)
parser.add_argument("--density", help=("Density function/kernel estimator."
                    " ('expset', 'gausset', 'setmax'; default: 'gausset')"),
                    default='gausset')
parser.add_argument("--bw", help=("Bandwidth/strinctness for the kernel estimator."
                    " (default: 5.0)"), type=float, default=5.0)
parser.add_argument("--hitmiss", help=("Number of samples to build the"
                                      " hit-and-missing topology (0 --> 25%% of"
                                      " '--sample'; default: 50)"),
                    type=int, default=50)
parser.add_argument("--njobs", help=("Number of cores to use for simulating"
                    " (default: -1 = all available cores)"), type=int, default=-1)
parser.add_argument("--bias", help=("Bias parameter for linear separator argument"
                    " densities (default: 1.0)"), type=float, default=1.0)
args = parser.parse_args()

input_oie = args.in_oie
input_plain = args.in_txt

NJOBS = args.njobs
SAMPLE_SIZE = args.sample  # eg. 100
N_STEPS = args.nsteps  # 120  #e.g.: n_input_oie/SAMPLE_SIZE=12167/100=121.67
ngramr = tuple(args.ngrams)  # (1, 3)
t_size = args.wsize  # 10
N_TRAJECTORIES = 2
analyzer = CountVectorizer(analyzer=ANALYZER, ngram_range=ngramr)\
                                .build_analyzer()

if args.hitmiss == 0:
    n_hit_miss = int(SAMPLE_SIZE * 0.25)  # 25% of the step sample size
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

if args.output is None:
    out_name = make_output_name(args)
else:
    out_name = args.output
    
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

compute_mi_steps(gsAkdf, prod_cols=TOANALYZE, out_csv="oie_" + out_name,
                    sigma=args.bw, density=args.density,
                    ngramr=ngramr, n_hit_miss=n_hit_miss, bias=args.bias)

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
                    out_csv="rdn_" + out_name,
                    sigma=args.bw,
                    ngramr=ngramr, n_hit_miss=n_hit_miss)
                    
logging.info("Terminated in {}s...".format(time.time() - t_start))
