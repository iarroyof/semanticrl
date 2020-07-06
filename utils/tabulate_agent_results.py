import pandas as pd
import sys
from pdb import set_trace as st

agent = 'oie'

try:
    results_incsv = sys.argv[1]
except:
    results_incsv = ("results/train_results_originalExtremaWeighted"
                            "GaussianBayesianExpBeta1e8Pvalues_rwd.csv")
try:
    results_outcsv = sys.argv[2]
except:
    results_outcsv = 'Results.csv'

def map_measures(m):
    mapp = {
        'HX_HY_HZ': 'Joint E.',
        'IXYZ_IYZX_IXZY': 'CMI',
        'IXY_IYZ_IXZ': 'MI',
        'HXY_HYZ_HXZ': 'Entropy'
    }
    return mapp[m]


df = pd.read_csv(results_incsv)
df_list = []
measures = df.Measure.unique()
measure_wise_params = {}
rdn_queries = {}

for pr in measures:
    dens_wise = pd.concat([
        df.query(
                "Measure=='{}' and Agent=='{}' and density=='expset'" \
              .format(pr, agent))
              .drop_duplicates() \
              .drop(['Reward', 'hitmiss', 'wsize', 'Agent'], axis=1) \
              .sort_values('pReward', ascending=False).head(5),
            df.query(
                "Measure=='{}' and Agent=='{}' and density=='gausset'"\
              .format(pr, agent)) \
              .drop_duplicates() \
              .drop(['Reward', 'hitmiss', 'wsize', 'Agent'], axis=1) \
              .sort_values('pReward', ascending=False).head(5)
          ])
    measure_wise_params[pr] = [] 
    measure_wise_params[pr].append(dens_wise[
        dens_wise.density=='expset'].drop(
            ['ABpvalue', 'BCpvalue', 'CApvalue', 'pReward'],
            axis=1).iloc[0].items())
    measure_wise_params[pr].append(dens_wise[
        dens_wise.density=='gausset'].drop(
            ['ABpvalue', 'BCpvalue', 'CApvalue', 'pReward'], 
            axis=1).iloc[0].items())

    measure_wise_params[pr][0] = ' and '.join([
        '=='.join([i, "'" + v + "'"] if isinstance(v, str)
            else [i, str(v)]) for i, v in measure_wise_params[pr][0]])
    measure_wise_params[pr][1] = ' and '.join([
        '=='.join([i, "'" + v + "'"] if isinstance(v, str)
            else [i, str(v)]) for i, v in measure_wise_params[pr][1]])
    df_list.append(dens_wise)

results_df = pd.concat(df_list)
results_df['Measure'] = results_df['Measure'].apply(map_measures)

formats = {'bias': '{:.1f}', 'bw': '{:.1f}',
            'pReward': '{:1.5f}', 'ABpvalue': '{:1.3E}',
            'BCpvalue': '{:1.3E}', 'CApvalue': '{:1.3E}'}

for col, f in formats.items():
    results_df[col] = results_df[col].map(lambda x: f.format(x))

results_df[["Measure", "Sample", "bias", "bw", "density",
	    "ngrams", "ABpvalue", "BCpvalue", "CApvalue", "pReward"]].to_csv(
						results_outcsv, index=False)

# For random agent
rdn_dfs = []
for m, qs in measure_wise_params.items():
    rdn_dfs.append(
        pd.DataFrame([
            dict(df.query(qs[0] + " and Agent=='rdn'") \
                    .drop(['Reward', 'hitmiss', 'wsize', 'Agent'], axis=1) \
                    .iloc[0].items()),
            dict(df.query(qs[1] + " and Agent=='rdn'") \
                    .drop(['Reward', 'hitmiss', 'wsize', 'Agent'], axis=1) \
                    .iloc[0].items())
            ])
        )

rdn_df = pd.concat(rdn_dfs)
for col, f in formats.items():
    rdn_df[col] = rdn_df[col].map(lambda x: f.format(x))
rdn_df['Measure'] = rdn_df['Measure'].apply(map_measures)
rdn_df[["Measure", "Sample", "bias", "bw", "density",
            "ngrams", "ABpvalue", "BCpvalue", "CApvalue", "pReward"]].to_csv(
				results_outcsv.replace('oie', 'rdn'), index=False)

