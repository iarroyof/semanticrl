import pandas as pd
import numpy as np
import re
import math
import argparse
from scipy.stats import norm as Normal
from os import walk
import sys
from joblib import Parallel, delayed

from pdb import set_trace as st

def semantic_reward(csv, cols, measure, sample, sort=True, sigma=1000.0):
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
    pars = list([tuple(i.split('-')) for i in pars.split('_')])
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
    sigma = df[mean_df.index[1]].std()
    l = abs(a - c) / 2
    dist = abs(b - l)
    try:
        z = abs(a - c) / (sigma * math.sqrt(2 * math.pi))
        line["Reward"] = z * math.exp(-dist ** 2/(2 * sigma ** 2))
    except:
        line["Reward"] = 0.0
        
    pvals = (   Normal(loc=b, scale=sigma).pdf(a),
                Normal(loc=c, scale=sigma).pdf(b),
                Normal(loc=a, scale=sigma).pdf(c)
            )
    line.update(zip(['ABpvalue',
                     'BCpvalue',
                     'CApvalue'], pvals))

    return line


results = []

for measure_type in ['h', 'cmi', 'mi', 'jh']:
    results_dir = "/almac/ignacio/results_srl_env/wsize-8"
    columns, posfijo = {'h':
        (["$H[h(Z, Z)]$", "$H[h(Y, Y)]$", "$H[h(X, X)]$"], "HX_HY_HZ"),
                   'cmi':
        (["$I[h(Y+Z, X)]$", "$I[h(X+Z, Y)]$", "$I[h(X+Y, Z)]$"], "IXYZ_IYZX_IXZY"),
                    'mi':
        (["$I[h(Y, X)]$", "$I[h(Z, X)]$", "$I[h(Z, Y)]$"], "IXY_IYZ_IXZ"),
                    'jh':
        (["$H[h(Y+Z, Y+Z)]$", "$H[h(X+Z, X+Z)]$", "$H[h(X+Y, X+Y)]$"], "HXY_HYZ_HXZ")
    }[measure_type]
    print("COMPUTING REWARDS FOR MEASURE: %s" % posfijo)
    samples = next(walk(results_dir))[1]
    for sample in samples:
        result_files = [f for f in next(walk('/'.join([results_dir, sample])))[2]
                        if (f.startswith("rdn_bias-") or f.startswith("oie_bias-"))
                        and ('bias' in f and 'hitmiss' in f and
                        'bw' in f and 'density' in f and 'ngrams' in f)
                    ]
        srwd = partial(semantic_reward, cols=columns,
                    measure=posfijo, sigma=1000.0, sample=sample)
        dicts = Parallel(n_jobs=-1, verbose=10)(
                delayed(srwd)('/'.join([results_dir, sample, file]))
                                                    for file in result_files)
        results += dicts

pd.DataFrame(results).to_csv("results/all_results_originalExtremaWeightedGaussian_rwd.csv",
                                index=False)
