import pandas as pd
import numpy as np
import re
import argparse
from pdb import set_trace as st


def semantic_reward(csv, comparing_cols, sort=True):
    df = pd.read_csv(csv) \
           .replace(np.inf, np.NaN) \
           .interpolate(method='linear')
    if not sort:
        a, m, b = df[comparing_cols].mean().values
        sigma = df[comparing_cols[1]].std()
    else:
        mean_df = df[comparing_cols].mean().sort_values()
        a, m, b = mean_df.values
        sigma = df[mean_df.keys()[1]].std()
    
    l = abs(a - b) / 2
    z = abs(a - b) / (sigma * np.sqrt(2 * np.pi))
    dist = abs(m - l)

    return z * np.exp(-dist ** 2/(2 * sigma ** 2))


parser = argparse.ArgumentParser()
parser.add_argument("--cols", help="Comparing columns to plot ( "
                    "spaced strings, being column names. Special characters "
                   "like the latex '$' must be scaped, e.g. '\$I(h(X, Y+Z))'.)",
                    type=str, nargs='+')
parser.add_argument("--in_csv", help="Input csv file name")

args = parser.parse_args()


reward = semantic_reward(csv=args.in_csv, comparing_cols=args.cols)
print(reward)
