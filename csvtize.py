#,file,reward
#27,/almac/ignacio/results_srl_env/wsize-8/sample-160/rdn_bias-0.0_hitmiss-1.0_bw-3.0_density-expset_ngrams-2t3.csv,3.1938901222099996
# 24,oie_bias-0.0_hitmiss-1.0_bw-3.0_density-expset_ngrams-1t2.csv,0.2993974183619815
# rewards_sorted_sample-52_oie_rewards_IXYZ_IYXZ_IZXY.csv   sorted_sample-60_rdn_rewards_HX_HY_HZ.csv.csv
import re
import sys
import pandas as pd


in_csv = sys.argv[1]
if in_csv.endswith('.csv.csv'): exit()
out_df = []
with open(in_csv) as f:
    wss = [
           ('sample', int(re.findall("sample-(\d+)_", in_csv)[0])),
           ('measure', re.findall("_rewards_(\w+).csv", in_csv)[0]), 
           ('wsize', 8)
          ]
    for l in f.readlines()[1:]:
        if l.endswith(','):
            fields = (l.strip() + "-0.0").split(',')
        else:
            fields = l.strip().split(',')
        if False in ['bias' in fields[1], 
                    'hitmiss' in fields[1],
                    'bw' in fields[1],
                    'density' in fields[1],
                    'ngrams' in fields[1]]:
            continue

        if '/' in fields[1]:
            pars = fields[1].split('/')[-1].split('csv')[0].strip('.')
        else:
            pars = fields[1].split('csv')[0].strip('.')
        pars = 'agent-' + pars
        pars = list([tuple(i.split('-')) for i in pars.split('_')])
        
        try:
            line = dict(wss + pars)
        except:
            st()
            line = dict(wss + [p if len(p) == 2 else (p[0], -0.0) for p in pars])
        line['reward'] = fields[2]
        line['file'] = fields[1]

        for i in line.items():
            try:
                line[i[0]] = int(i[1])
            except ValueError:
                try:
                    line[i[0]] = float(i[1])
                except ValueError:
                    pass

        out_df.append(line)

pd.DataFrame(out_df).to_csv(in_csv + ".csv", index=False)
