import pandas as pd
import numpy as np
import re
import argparse
from pdb import set_trace as st
from scipy.stats import ttest_ind
from scipy.stats import norm as Normal


def semantic_reward(csv, comparing_cols, sort=True):
    df = pd.read_csv(csv) \
           .replace(np.inf, np.NaN) \
           .interpolate(method='linear')
    ###pvals = []
    if not sort:
        a, m, b = df[comparing_cols].mean().values
        sigma = df[comparing_cols[1]].std()
    else:
        #df = (df - df.mean())/df.std()
        #df = (df - df.min())/(df.max() - df.min())
        mean_df = df[comparing_cols].mean().sort_values()
        #a, m, b = mean_df.values
        a, m, b = ((mean_df - mean_df.min())
                        /(mean_df.max() - mean_df.min())).values
        #sigma = df[mean_df.keys()[1]].std()
        #sigma = min(df[comparing_cols].std())
        sigma = 0.15
    #st()    
    l = abs(a - b) / 2
    #z = (sigma * np.sqrt(2 * np.pi)) ** (-1.0)
    ##z = abs(a - b) / (sigma * np.sqrt(2 * np.pi))
    #dist = -np.log(abs(m - l))
    #return z * np.exp(-dist ** 2/(2 * sigma ** 2))
    return Normal(loc=l, scale=sigma).cdf(m)
    ###pvals.append(1 - ttest_ind(df[comparing_cols[0]], df[comparing_cols[1]])[1])
    ###pvals.append(1 - ttest_ind(df[comparing_cols[1]], df[comparing_cols[2]])[1])
    ###pvals.append(1 - ttest_ind(df[comparing_cols[2]], df[comparing_cols[0]])[1])

    ###return min(pvals)

parser = argparse.ArgumentParser()
parser.add_argument("--cols", help="Comparing columns to plot ( "
                    "spaced strings, being column names. Special characters "
                   "like the latex '$' must be scaped, e.g. '\$I(h(X, Y+Z))'.)",
                    type=str, nargs='+')
parser.add_argument("--in_csv", help="Input csv file name")

args = parser.parse_args()

try:
    reward = semantic_reward(csv=args.in_csv, comparing_cols=args.cols)
except TypeError:
    reward = 0.0
print(reward)
