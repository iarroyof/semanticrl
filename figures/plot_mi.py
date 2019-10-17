import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rc
matplotlib.use
import matplotlib.pyplot as plt
plt.style.use('ggplot')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
import argparse
from pdb import set_trace as st


def plot_mis(csv, comparing_cols, title, roll=20):
    df = pd.read_csv(csv) \
           .replace(np.inf, np.NaN) \
           .interpolate(method='linear')
    colors = ['b','g','r','c','m','y','k','w','burlywood']
    plt.figure()
    ax = plt.subplot(111)
    for column, color in zip(comparing_cols, colors):
        col = df[column]
        ma = col.rolling(roll).mean()
        mstd = col.rolling(roll).std()
        plt.plot(ma.index, ma, color, label=column)
        plt.fill_between(mstd.index, ma - 2 * mstd, ma + 2 * mstd,
                     color=color, alpha=0.2)
        plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Bits')
        ax.legend()
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument("--cols", help="Comparing columns to plot (semicolon "
					"spaced strings, being column names. Special characters "
                    "like the latex '$' must be scaped, e.g. '\$'.)",
                    type=str, nargs='+')
parser.add_argument("--in_csv", help="Input csv file name")
parser.add_argument("--title", help="Title of the figure")
parser.add_argument("--roll", help="Rolling mean window (default: 20).",
                    type=int)
args = parser.parse_args()

plot_mis(csv=args.in_csv, comparing_cols=args.cols, title=args.title,
        roll=args.roll)
