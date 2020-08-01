import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
import sys

from pdb import set_trace as st

def map_measures(m):
    mapp = {
        'HX_HY_HZ': 'JE',
        'IXYZ_IYZX_IXZY': 'CMI',
        'IXY_IYZ_IXZ': 'MI',
        'HXY_HYZ_HXZ': 'Entropy'
    }
    return mapp[m]


infile = sys.argv[1]
#df = pd.read_csv("results/train_results_originalExtremaWeighted"
#                "GaussianBayesianExpBeta1e8Pvalues_rwd.csv")
drop_rwd = 'Reward'
nng = 1
ylog = False
palette = 'Dark2' #'tab10'
ptte_start = 3
gauss_start = 3

df = pd.read_csv(infile)

df['Measure'] = df['Measure'].apply(map_measures)
colors = plt.get_cmap(palette).colors
rwdt = 'pReward' if drop_rwd == 'Reward' else 'Reward'
measures = df.Measure.unique()
categories = ['Agent',
              'ngrams']
numerics = ['Sample', 'bias', 'bw']
to_drop = ['wsize', drop_rwd, 'hitmiss',
           'ABpvalue', 'BCpvalue', 'CApvalue']

df = df.drop(to_drop, axis=1)

fig, axs = plt.subplots(measures.shape[0], len(numerics), figsize=(8, 20))
for density in ['expset', 'gausset']:
  for i, mea in enumerate(measures):
    
    best_params = dict(
        df[(df.Measure==mea) &
           (df.Agent=='oie') &
           (df.density==density)].sort_values(
                rwdt, ascending=False).drop(
                    [rwdt, 'Measure'], axis=1).iloc[0].items())
   
    print(f"Best hyperparams for {mea}: {best_params}")

    rank_ngrams = df[
        (df.Measure==mea) &
        (df.Agent=='oie') &
        (df.density==density)].sort_values(
            rwdt, ascending=False).drop(
                [rwdt, 'Measure'], axis=1).ngrams

    rank_ngrams = list(dict.fromkeys(rank_ngrams).keys())
    n_grams = list(set(rank_ngrams))
    best_ngrams = []
    for n in rank_ngrams:
        if n in best_ngrams:
            continue
        else:
            try:
                best_ngrams.append(n)
                n_grams.remove(n)
            except:
                break
    if nng == 2:
        best_ngrams = [best_ngrams[0], best_ngrams[-1]]
    elif nng == 1:
        best_ngrams = [best_ngrams[0]]
    elif nng > 2:
        best_ngrams = best_ngrams[:nng]
    
    mdf = df[(df.Measure==mea) & (df.density==density)].drop('Measure' , axis=1)
    axs[i, 0].set_ylabel(mea)
    for j, num in enumerate(numerics):
        color = ptte_start if density.startswith('e') else ptte_start + gauss_start
        if ylog:
            axs[i, j].axes.set_yscale('log', nonposy='clip')
        
        bd = best_params.copy()
        bd.pop(num)
        [bd.pop(c) for c in categories]
        ind = [True] * len(mdf)
        for n, v in bd.items():
            ind = ind & (mdf[n] == v)
        _mdf = mdf[ind]

        for c in categories:    
            ind = [True] * len(_mdf)
            cd = best_params.copy()
            cd.pop(c)
            cd.pop(num)
            for n, v in cd.items():
                ind = ind & (_mdf[n] == v)
            
            cdf = _mdf[ind].sort_values(num)
            if c == 'ngrams':
                iter_cv = best_ngrams
            elif c == 'Agent':
               iter_cv = ['rdn']
            else:
                iter_cv = cdf[c].unique()
                
            for cv in iter_cv:
                if nng == 1 and c == 'ngrams':
                    nam0 = 'Agent'
                    nam1 = 'oie'
                else:
                    nam0 = c
                    nam1 = cv
                    
                #if num == 'bw': st()
                cdf[cdf[c]==cv].plot(num, rwdt,
                    label=nam0 + '_' + nam1 + '_' + density,
                    ax=axs[i, j],
                    color=colors[color])

                if j != len(axs[0, :]) - 1:
                    try:
                        axs[i, j].get_legend().remove()
                    except AttributeError:
                        axs[i, j].legend().remove()
                else:
                    axs[i, j].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                color += 1

plt.show()
            
