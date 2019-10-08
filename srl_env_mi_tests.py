
from srl_env_v01 import textEnv
import itertools
import random
import numpy as np
import pandas as pd
import re
import math
from jellyfish import damerau_levenshtein_distance as dlbsh
import logging
import argparse
import string

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from joblib import Parallel, delayed
    from sklearn.feature_extraction.text import CountVectorizer

from pdb import set_trace as st

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)

ANALYZER = 'char'


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
        distance = dot_distance(S, M, binary=True,
                                            euclid=False, ngramr=ngramr)
    elif metric == 'euc':
        distance = dot_distance(S, M, binary=False,
                                            euclid=True, ngramr=ngramr)

    if distance <= 1.0:
        return 1.0 / (np.sqrt(2 * np.pi * sigma ** 2))
    else:
        return (1.0 / (np.sqrt(2 * np.pi * sigma ** 2))) * math.exp(
                                             -distance ** 2 / (2 * sigma ** 2))


def compute_set_probability(Akdf, hit_miss_samples=50, sigma=5.0,
                            cols=['X', 'Y', 'Z'], metric='hmm', ngramr=(1, 3)):
    try:
        assert len(Akdf.index) >= hit_miss_samples, (
                                                "The number of perimeter"
                                                " samples must be less or equal"
                                                " than the number of samples")
    except AssertionError:
        return None

    deps = []
    for a, b in itertools.product(*[cols] * 2):
        if not ((a, b) in deps and (b, a) in deps):
            deps.append((a, b))
            Akdf['+'.join((a, b))] = Akdf[[a, b]].apply(lambda x: ' '.join(x),
                                                          axis=1)
    prod_cols = []
    for a, b in itertools.product(*[cols + ['Y+Z', 'X+Z', 'X+Y']] * 2):
        if not ((a, b) in prod_cols or (b, a) in prod_cols):
            prod_cols.append((a, b))
            measures = []
            for _ in range(hit_miss_samples):
                B = Akdf[b].sample(frac=1)
                measures.append(
                    np.vectorize(set_valued_gaussian)(
                        Akdf[a].str.lower(), B.str.lower(), sigma=sigma,
                        metric=metric, ngramr=ngramr)
                )
            joincol = '$N_\sigma\{h(' + ', '.join((a, b)) + ')\}$'
            Akdf[joincol] = np.vstack(measures).mean(axis=0)
    return Akdf.dropna()


def rdn_partition(state):
    tokens = state.split()
    idxs = sorted(random.sample(list(range(1, len(tokens) - 1)), 2))
    return {'X': " ".join(tokens[:idxs[0]]),
            'Y': " ".join(tokens[idxs[0]:idxs[1]]),
            'Z': " ".join(tokens[idxs[1]:])}


def compute_mutuals(df, cols):

    patt = r'N_\\sigma\\{h\(([XYZ]\+?[XYZ]?), ([XYZ]\+?[XYZ]?)\)'
    pairs = sum([re.findall(patt, c) for c in cols], [])
    selfs = ["$N_\sigma\{h(" + ', '.join(p) + ")\}$"
                    for p in pairs if p[0] == p[1]]
    joins = [("$N_\sigma\{h(" + ', '.join(p) + ")\}$",
              "$N_\sigma\{h(" + ', '.join([p[1], p[1]]) + ")\}$",
              "$N_\sigma\{h(" + ', '.join([p[0], p[0]]) + ")\}$")
                    for p in pairs if p[0] != p[1]]
    entropy = lambda x: -x * np.log2(x) if x > 0.0 else 0.0
    centropy = lambda x: -x[1] * np.log2(x[1] / x[0]) \
                            if x[0] > 0 and x[1] > 0 else 0.0
    for s in selfs:
        try:
            icol = "$H[h(" + ', '.join(re.findall(patt, s)[0]) + "')]$"
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


def compute_mi_steps(Akdf, out_csv, metric='hmm', sigma=5.0,
                                              n_hit_miss=50, ngramr=(1, 3)):
    """ compute_set_probability(Akdf, hit_miss_samples=50, sigma=5.0,
                                cols=['X', 'Y', 'Z'], metric='hmm'
    """
    A_tau = [Akdf[i:i + SAMPLE_SIZE]
                for i in range(0, N_STEPS * SAMPLE_SIZE, SAMPLE_SIZE)]

    logging.info(f"Computing probabilities of random sets for {N_STEPS} steps.")
    Psim_Aks = Parallel(n_jobs=NJOBS)(
                    delayed(compute_set_probability)(
                        A_k, hit_miss_samples=n_hit_miss,
                                     metric=metric, sigma=sigma, ngramr=ngramr)
                                                            for A_k in A_tau)
    probcs = [
        '$N_\sigma\{h(X, Y)\}$', '$N_\sigma\{h(Y, Z)\}$',
        '$N_\sigma\{h(X, Z)\}$',
        '$N_\sigma\{h(X, X)\}$', '$N_\sigma\{h(Y, Y)\}$',
        '$N_\sigma\{h(Z, Z)\}$',
        '$N_\sigma\{h(X, Y+Z)\}$', '$N_\sigma\{h(Y, X+Z)\}$',
        '$N_\sigma\{h(Z, X+Y)\}$']
    info_steps = Parallel(n_jobs=NJOBS)(
                    delayed(compute_mutuals)(df, probcs)
                                           for df in Psim_Aks if not df is None)
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
parser.add_argument("--ngrams", help="N-gram range for sklearn ngramer",
                    type=int, nargs='+')
parser.add_argument("--wsize", help=("Window size for text environment states"
                                     " (default: 10)"),
                    type=int, default=10)
parser.add_argument("--metric", help=("Metric of the measure space (default:"
                                      " 'hmm' = Hamming)"),
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
parser.add_argument("--bw", help=("Bandwidth for the kernel estimator."
                                      " (default: 5.0)"),
                    type=float, default=5.0)
parser.add_argument("--hitmiss", help=("Number of samples to build the"
                                      " hit-and-missing topology (default: 50)"),
                    type=int, default=50)
parser.add_argument("--njobs", help=("Number of cores to use for simulating"
                                     " (default: -1)"),
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
cols = ['X', 'Y', 'Z']
N_TRAJECTORIES = 2
perimeter = int(SAMPLE_SIZE * 0.25)  # 25% of the sample size

logging.info("Reading input file '{}'".format(input_oie))
# Randomize the input gold standard
gsAkdf = pd.read_csv(input_oie, delimiter='\t', keep_default_na=False,
                     names=['score'] + cols)[cols].sample(frac=1.0)
# Remove triplets with only punctuation
for c in cols:
    gsAkdf[c] = gsAkdf[c].apply(clean)

gsAkdf = gsAkdf[~gsAkdf['X'].isin(['', ' '])
                & ~gsAkdf['Y'].isin(['', ' '])
                & ~gsAkdf['Z'].isin(['', ' '])]
# Take N_STEPS and compute their marginal and joint informations
logging.info("Computing MI for OpenIE actions...")
compute_mi_steps(gsAkdf, input_oie.split('.')[0],
                    metric=args.metric, sigma=args.bw,
                    ngramr=ngramr, n_hit_miss=args.hitmiss)

# Create text environment sampler
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
compute_mi_steps(rdn_Akdf, input_plain)
logging.info("Terminated...")
