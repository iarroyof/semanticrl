import pandas as pd
import matplotlib.pyplot as plt 


from pdb import set_trace as st

df = pd.read_csv("results/all_results_originalExtremaWeightedGaussianBayesianExpBeta1e8Pvalues_rwd.csv")

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',   
             'tab:brown', 'tab:pink']
drop_rwd = 'Reward'
rwdt = 'pReward' if drop_rwd == 'Reward' else 'Reward'
measures = df.Measure.unique()
categories = ['Agent', 'density', 'ngrams']
numerics = ['Sample', 'bias', 'bw']
to_drop = ['wsize', drop_rwd, 'hitmiss',
           'ABpvalue', 'BCpvalue', 'CApvalue']
nng = 2
ylog = False
df = df.drop(to_drop, axis=1)
fig, axs = plt.subplots(measures.shape[0], len(numerics))
#fig, axs = plt.subplots(2, 2)

for i, mea in enumerate(measures):
    best_m = dict(df[df.Measure==mea].sort_values(
                            rwdt, ascending=False).drop(
                                [rwdt, 'Measure'],
                             axis=1).iloc[0].items()
                  )
    rank_ngrams = df[df.Measure==mea].sort_values(
                            rwdt, ascending=False).drop(
                                [rwdt, 'Measure'],
                             axis=1).ngrams
    best_ngrams = list(dict.fromkeys(rank_ngrams).keys())[:nng]
    mdf = df[df.Measure==mea].drop('Measure' , axis=1)
    axs[i, 0].set_ylabel(mea)
    for j, num in enumerate(numerics):
        if ylog:
            axs[i, j].set_yscale('log')
        ind = [True] * len(mdf)
        bd = best_m.copy()
        bd.pop(num)
        [bd.pop(c) for c in categories]
        for n, v in bd.items():
            ind = ind & (mdf[n] == v)
        _mdf = mdf[ind]
        color = 0
        for c in categories:    
            ind = [True] * len(_mdf)
            cd = best_m.copy()
            cd.pop(c)
            cd.pop(num)
            for n, v in cd.items():
                ind = ind & (_mdf[n] == v)
            
            cdf = _mdf[ind].sort_values(num)
            if c == 'ngrams':
                iter_cv = best_ngrams
            else:
                iter_cv = cdf[c].unique()
            
            for cv in iter_cv:
                cdf[cdf[c]==cv].plot(num, rwdt,
                    label=c + '_' + cv, ax=axs[i, j], color=colors[color])
                color += 1
                
fig.set_figheight(50)
fig.set_figwidth(20)
plt.show()
            
