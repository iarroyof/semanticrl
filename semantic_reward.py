import pandas as pd
import numpy as np
import re
import argparse
from pdb import set_trace as st


def semantic_reward(csv, comparing_cols):
    df = pd.read_csv(csv) \
           .replace(np.inf, np.NaN) \
           .interpolate(method='linear')
#    st()
    a, m, b = df[comparing_cols].mean().values
    _, sigma, _ = df[comparing_cols].std().values    
    l = abs(a - b) / 2
    z = abs(a - b) / (sigma * np.sqrt(2 * np.pi))
    dist = abs(m - l)
    return z * np.exp(-dist ** 2/(2 * sigma ** 2))
    #for column in comparing_cols:
    #    col = df[column]
        #ma = col.rolling(roll).mean()
        #mstd = col.rolling(roll).std()


parser = argparse.ArgumentParser()
parser.add_argument("--cols", help="Comparing columns to plot ( "
                    "spaced strings, being column names. Special characters "
                   "like the latex '$' must be scaped, e.g. '\$I(h(X, Y+Z))'.)",
                    type=str, nargs='+')

parser.add_argument("--in_csv", help="Input csv file name")

args = parser.parse_args()


#columns = re.sub(r'\$\s+\$', '\$__\$', args.cols) \
#            .replace('\\', '').split("__")
#assert len(columns) != 3  # Provide exactly three column names
#st()
reward = semantic_reward(csv=args.in_csv, comparing_cols=args.cols)
print(reward)
