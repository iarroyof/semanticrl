
import itertools
import random, csv
import numpy as np
import pandas as pd
from dask import dataframe as dd
import tensorflow as tf
import os, re, random, math
import matplotlib
matplotlib.use
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import logging
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from joblib import Parallel, delayed
from jellyfish import damerau_levenshtein_distance as dlbsh
from scipy.spatial.distance import directed_hausdorff as hsdff
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                                        level=logging.INFO)

fitting_sim_oie = "data/sim_train.txt.oie"
develop_sim_oie = "data/sim_test.txt.oie"
fitting_unr_oie = "data/dis_train.txt.oie"
develop_unr_oie = "data/dis_test.txt.oie"

fitting_sim = "data/sim_train_.txt"
develop_sim = "data/sim_test_.txt"
fitting_unr = "data/dis_train_.txt"
develop_unr = "data/dis_test_.txt"

SAMPLE_SIZE = 15
N_STEPS = 50
dev_sample = 0.30
perimeter = 10
ngramr = (1, 3)
cols=['X', 'Y', 'Z']



def intersect(r, s):
        return set(r).intersection(s)

def unify(r, s):
        return set(r).union(s)

def ref_jaccard(sizes, item):
    try:
        return [i/sizes[item] for i in sizes]
    except ZeroDivisionError:
        return 0

def lev_hausdorff(A, B):
    #return max([min([dlbsh(a, b) for b in B]) for a in A])
    D = {}
    h = 0
    for a in A:
        shortest = np.inf
        for b in B:
            D[(a, b)] = dlbsh(a, b)
            if D[(a, b)] < shortest:
                shortest = D[(a, b)]
        if shortest > h:
            h = shortest

    return h

    
def dot_distance(A, B, binary=True, euclid=False):
    try:
        assert isinstance(A, str) and isinstance(B, str)
    except AssertionError:
        if np.nan in [A, B]:
            return 10000.0
        
    vectorizer = CountVectorizer(binary=binary, analyzer='char',
                                 ngram_range=ngramr)
    
    X = vectorizer.fit_transform([A, B])
    if euclid:
        return (X[0] - X[1]).dot((X[0] - X[1]).T).sum()
    else:
        return (X[0].toarray() ^ X[1].toarray()).sum()
    
def euc_hausdorff(A, B):
    discret = LabelEncoder()

    hausdorff = lambda u, v: max(hsdff(u, v)[0], hsdff(v, u)[0])
    discret.fit(A + B)
    a = discret.transform(A)
    b = discret.transform(B)
    if len(a) - len(b) < 0:
        a = a.tolist() + [-1] * abs(len(a) - len(b))
    elif len(a) - len(b) > 0:
        b = b.tolist() + [-1] * (len(a) - len(b))

    return hausdorff(np.array(a).reshape(1, -1), np.array(b).reshape(1, -1))
    
def set_valued_gaussian(S, M, sigma=1.0, metric='h'):
    if metric == 'h':
        if not(isinstance(S, list) and isinstance(M, list)):
            S = ngramer(S)
            M = ngramer(M)
        #distance = cat_hausdorff(S, M)
        distance = lev_hausdorff(S, M)
    elif metric == 'l':
        assert isinstance(S, str) and isinstance(M, str)
        distance = dlbsh(S, M)
    elif metric == 'hmm':
        distance = dot_distance(S, M, binary=False, euclid=True )
    #print(distance)
    if distance == 0:
        return 1.0
    else:
        return (1/(np.sqrt(2 * np.pi * sigma ** 2))) * math.exp(
                                            -distance ** 2/(2 * sigma ** 2))

def computeNab(df, ca, cb):
    list(map(intersect, df[[ca, cb]]))


def compute_set_probability(Akdf, perimeter_samples=50, sigma=5.0,
                                                        cols=['X', 'Y', 'Z']):
    
    assert len(Akdf.index) >= perimeter_samples, ("The number of perimeter" 
                                                "samples must be less or equal"
                                                "than the number of samples")
        
    v = TfidfVectorizer(analyzer='char', ngram_range=ngramr)
    ngramer = v.build_analyzer()
    m = perimeter_samples
    Akddf = dd.from_pandas(Akdf, npartitions=5)  # Convert to Dask
    deps = []
    for a, b in itertools.product(*[cols] * 2):
        if not ((a, b) in deps and (b, a) in deps):
            deps.append((a, b))
            Akddf['+'.join((a, b))] = Akddf[[a, b]].apply(lambda x: ' '.join(x),
                                                          axis=1, meta=('str'))
    prod_cols = []
    for a, b in itertools.product(*[cols + ['Y+Z', 'X+Z', 'X+Y']] * 2):
        if not ((a, b) in prod_cols or (b, a) in prod_cols):
            prod_cols.append((a, b))
            measures = []
            for _ in range(m):
                B = Akddf[b].sample(frac=1)
                measures.append(
                    np.vectorize(set_valued_gaussian)(
                        Akddf[a].str.lower(), B.str.lower(), sigma=sigma,
                        metric='hmm')
                )
                                                            
            Akddf['P{hamming(' + ', '.join((a, b)) + ')}'] = dd.from_array(
                                            np.vstack(measures).mean(axis=0))

    return Akddf.compute()


v = TfidfVectorizer(analyzer='char', ngram_range=ngramr)
ngramer = v.build_analyzer()
logging.info("Reading input file '{}'".format(fitting_sim_oie))
gsAkdf = pd.read_csv(fitting_sim_oie, delimiter='\t',
                            names=['score'] + cols)[cols]
dev_sim_gs_Akdf = gsAkdf.sample(frac=dev_sample)
dev_sim_gs_Akdf.shape

start = 0
end = SAMPLE_SIZE
dev_Psim_gsAks = []
A_tau = []
for s in range(N_STEPS):
#    dev_Psim_gsAks.append(compute_set_probability(
#                            dev_sim_gs_Akdf[start:end],
#                            perimeter_samples=perimeter))
    A_tau.append(dev_sim_gs_Akdf[start:end])
    #dev_Psim_gsAks[-1].head()
    end = end + SAMPLE_SIZE
    start = end
logging.info("Computing probabilities of random sets...")
dev_Psim_gsAks = Parallel(n_jobs=-1)(delayed(compute_set_probability)(
                                    A_k, perimeter_samples=perimeter)
                                        for A_k in A_tau)

#dev_Psim_gsAkdf = compute_set_probability(dev_sim_gs_Akdf,
#                                          perimeter_samples=perimeter)
#dev_Psim_gsAkdf = dev_Psim_gsAkdf.dropna()
#print(f"Fit sample size: {len(dev_Psim_gsAkdf.index)}")
#fit_sim_gs_Akdf = gsAkdf.drop(dev_sim_gs_Akdf.index)
#fit_Psim_gsAkdf = compute_set_probability(fit_sim_gs_Akdf,
#                                          perimeter_samples=perimeter)
#fit_Psim_gsAkdf = fit_Psim_gsAkdf.dropna()
#print(f"Dev sample size: {len(fit_Psim_gsAkdf.index)}")
