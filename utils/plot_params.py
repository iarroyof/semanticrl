import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

from pdb import set_trace as st

def map_measures(m):
    mapp = {
        'HX_HY_HZ': 'JE',
        'IXYZ_IYZX_IXZY': 'CMI',
        'IXY_IYZ_IXZ': 'MI',
        'HXY_HYZ_HXZ': 'Entropy'
    }
    return mapp[m]
    

df = pd.read_csv("results/train_results_originalExtremaWeighted"
                "GaussianBayesianExpBeta1e8Pvalues_rwd.csv")

df['Measure'] = df['Measure'].apply(map_measures)
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',   
             'tab:brown', 'tab:pink']
drop_rwd = 'Reward'
rwdt = 'pReward' if drop_rwd == 'Reward' else 'Reward'
measures = df.Measure.unique()
categories = ['Agent',
              #'density',
              'ngrams']
df = df[df.density=='expset']
numerics = ['Sample', 'bias', 'bw']
to_drop = ['wsize', drop_rwd, 'hitmiss',
           'ABpvalue', 'BCpvalue', 'CApvalue']
nng = 2
ylog = False
df = df.drop(to_drop, axis=1)

fig, axs = plt.subplots(measures.shape[0], len(numerics), figsize=(8, 20))
#fig.set_figheight(50)
#fig.set_figwidth(20)

for i, mea in enumerate(measures):
    best_params = dict(
        df[(df.Measure==mea) & (df.Agent=='oie')].sort_values(
            rwdt, ascending=False).drop(
                [rwdt, 'Measure'], axis=1).iloc[0].items())
    rank_ngrams = df[(df.Measure==mea) & (df.Agent=='oie')].sort_values(
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
    #best_ngrams = best_ngrams[:nng]
    best_ngrams = [best_ngrams[0], best_ngrams[-1]]
    
    mdf = df[df.Measure==mea].drop('Measure' , axis=1)
    axs[i, 0].set_ylabel(mea)
    for j, num in enumerate(numerics):
        if ylog:
            axs[i, j].set_yscale('log')
        
        bd = best_params.copy()
        bd.pop(num)
        [bd.pop(c) for c in categories]
        ind = [True] * len(mdf)
        for n, v in bd.items():
            ind = ind & (mdf[n] == v)
        _mdf = mdf[ind]
        color = 0
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
                #if cv == '2t3': st()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    cdf[cdf[c]==cv].plot(num, rwdt,
                        label=c + '_' + cv, ax=axs[i, j], color=colors[color])
                    if len(w) == 1:
                        if w[0].category is UserWarning:
                            axs[i, j].plot(
                                cdf[cdf[c]==cv][num],
                                cdf[cdf[c]==cv]['pReward'].values + 10e-200,
                                color=colors[color])
                if j != len(axs[0, :]) - 1:
                    axs[i, j].get_legend().remove()
                else:
                    axs[i, j].legend(loc='center left', bbox_to_anchor=(1, 0.5))
                color += 1

plt.show()
            
