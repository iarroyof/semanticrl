import pandas as pd 
import sys

from pdb import set_trace as st
rewards_f = sys.argv[1]

# rewards_f = "/almac/ignacio/results_srl_env/wsize-8/sample-310/rewards_oie.csv"
df = pd.read_csv(rewards_f, sep='\t', names=["file", "reward"])

out = "_".join(rewards_f.split('/')[-2:])

df.sort_values('reward', ascending=False, inplace=True)

df.to_csv("results/rewards_sorted_" + out)
