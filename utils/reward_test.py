from functools import partial
from srl_env_v01 import textEnv
from scipy.stats import norm as Normal
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
import gym
from gym import spaces
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from joblib import parallel_backend
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import time
import warnings


STRUCTCOLS = ['X', 'Y', 'Z']

def rdn_partition(self, state):
    tokens = state.split()
    idxs = sorted(random.sample(list(range(1, len(tokens) - 1)), 2))
    action = np.split(tokens, idxs)
    return {c: " ".join(a) for c, a in zip(STRUCTCOLS, action)}



class LRSEntropyBasedTextSemanticsEnv(gym.Env):

    def __init__(self, input_text, sample=100, wsize=10, density='expset',
                 nsteps=200, tokenizer='char', njobs=-1, verbose=False):
        super(LRSEntropyBasedTextSemanticsEnv, self).__init__()

        self.input_plain    = input_text
        self.njobs          = njobs
        self.n_steps        = nsteps
        self.wsize          = wsize
        self.sample         = sample
        self.density        = density
        self.sigma          = sigma
        self.ngramr                             ngramr=(1, 3), bias=1.0
        self.tokenizer      = tokenizer
        self.verbose        = verbose
        

    def zlog(self, p):
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                return -p[1] * np.log2(p[1] / p[0])
            except Warning as e:
                if self.verbose:
                    logging.warning("Warningx: %s" % e)
                return 0.0
            except ZeroDivisionError as e:
                if self.verbose:
                    logging.warning("Errorx: %s" % e)
                return 0.0


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
                                      np.sqrt(sigma / np.pi) * math.exp(
                                          -sigma * (
                                            metric(filter_trap, hm) + bias)**2)
                                                  for hm in hit_miss])
        return probability(s)


    def compute_set_probability(self, Akdf, prod_cols, hit_miss_samples=50, sigma=5.0, 
                            ngramr=(1, 3), bias=1.0):
        try:
            assert len(Akdf.index) >= hit_miss_samples, (
                                                "The number of hit-and-miss"
                                                " samples must be less or equal"
                                                " than the number of samples")
        except AssertionError:
            return None

        if self.density == 'gausset':
            capacity = partial(self.gausset, ngramr=ngramr, sigma=sigma, bias=bias)
        elif self.density == 'setmax':
            capacity = partial(self.setmax, ngramr=ngramr, sigma=sigma)
        elif self.density == 'expset':
            capacity = partial(self.expset, ngramr=ngramr, sigma=sigma, bias=bias)
        else:
            assert density in ['gausset', 'setmax', 'expset']  # Unknown density

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
        """
        This method calls compute_set_probability() and compute_mutuals() 
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

        logging.info(
                f"Computing probabilities of random sets for {self.n_steps} steps.")
        with parallel_backend('multiprocessing' if BACKEND == 'mp' else 'loky'):
            t = time.time()
            P_Aks = Parallel(n_jobs=self.njobs)(
                        delayed(self.compute_set_probability)(
                            A_k, prod_cols=prod_cols, hit_miss_samples=n_hit_miss,
                            density=density, bias=bias, sigma=sigma, ngramr=ngramr)
                                                                    for A_k in A_tau)
            if self.verbose:
                logging.info("Estimated set probabilities in {}s..." \
                            .format(time.time() - t))

        with parallel_backend('multiprocessing' if BACKEND == 'mp' else 'loky'):
            t = time.time()
            info_steps = Parallel(n_jobs=self.njobs)(
                            delayed(self.compute_mutuals)(df, probcs)
                                for df in P_Aks if not (df is None or df.empty))
            if self.verbose:
                logging.info("Estimated MIs in {}s..." \
                                .format(time.time() - t))

        pd.DataFrame(info_steps).to_csv(out_csv)
        if os.path.exists(out_csv):
            logging.info("Output csv saved to {}".format(out_csv))
        else:
            logging.warning("Output csv {} wasn't saved!".format(out_csv))


    def _clean(self, x):
        translator = str.maketrans('', '', string.punctuation)
        try:
            return x.translate(translator)
        except AttributeError:
            return "__NULL__"


    def _formatf(self, param, decs=1):
        if isinstance(param, float)
:
            if param >= 1.0 or param == 0.0:
                f = "{:." + str(decs) + "f}"
            else:
                f = "{:." + str(decs) + "E}"
            return f.format(param)
        elif isinstance(param, str):
            return param




    def semantic_reward(csv, cols, measure, sample, beta=1e8):
        df = pd.read_csv(csv) \
               .replace(np.inf, np.NaN) \
               .interpolate(method='linear')
        wss = [
           ('Sample', int(sample.split('-')[1])),
           ('Measure', measure),
           ('wsize', 8)
          ]
        if '/' in csv:
            pars = csv.split('/')[-1].split('csv')[0].strip('.')
        else:
            pars = csv.split('csv')[0].strip('.')
        pars = 'Agent-' + pars
        pars = [
            tuple(i.split('-')) if len(i.split('-')) <= 2
                else (i.split('-')[0], '-'.join(i.split('-')[1:]))
                for i in pars.split('_')
        ]
        try:
            line = dict(wss + pars)
        except ValueError:
            pars.remove(('',))
            line = dict(wss + pars)

        for i in line.items():
            try:
                line[i[0]] = int(i[1])
            except ValueError:
                try:
                    line[i[0]] = float(i[1])
                except ValueError:
                    pass

        mean_df = df[cols].mean().sort_values()
        a, b, c = mean_df.values
        sa, sb, sc = df[mean_df.index].std().values
        l = abs(a - c) / 2
        dist = abs(b - l)
        try:
            z = abs(a - c) / (sb * math.sqrt(2 * math.pi))
            line["Reward"] = z * math.exp(-dist ** 2/(2 * sb ** 2))
        except:
            line["Reward"] = 0.0
        
        pvals = (   Normal(loc=a, scale=sa).pdf(b),
                    Normal(loc=b, scale=sb).pdf(c),
                    Normal(loc=c, scale=sc).pdf(a)
                )
        line.update(zip(['ABpvalue',
                     'BCpvalue',
                     'CApvalue'], pvals))
        line["pReward"] = math.exp(-beta * max(pvals))

        return line

