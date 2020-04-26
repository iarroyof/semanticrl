import pandas as pd 
import sys


s = sys.argv[1]
df = pd.read_csv("/almac/ignacio/results_srl_env/wsize-8/sample-" + s
                  + "/oie_rewards.csv", sep='\t', names=["file", "reward"])
df.sort_values(by=['reward'], ascending=False, inplace=True)
df.to_csv("oie_rewards_sorted_sample_" + s + ".csv")
