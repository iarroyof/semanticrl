RESULTS=/almac/ignacio/results_srl_env/wsize-8
AGENT="rdn"
POSFIJO=IXYZ_IYXZ_IZXY
for d in `ls $RESULTS`;
do
    ls --color=auto $RESULTS/${d}/${AGENT}_* | \
    parallel -k \
        'python semantic_reward.py --cols "\$I[h(Y+Z, X)]$" "\$I[h(X+Z, Y)]$" "\$I[h(X+Y, Z)]$" --in_csv {}' \
    > rs_${d}.txt; ls --color=auto $RESULTS/${d}/${AGENT}_* > files_${d}.txt;
    paste files_${d}.txt rs_${d}.txt \
    > $RESULTS/${d}/${AGENT}_rewards_${POSFIJO}.csv;
    python sort_rewards.py $RESULTS/${d}/${AGENT}_rewards_${POSFIJO}.csv;
done
#sorted_sample-110_rdn_rewards_HX_HY_HZ.csv
## "\$H[h(Z, Z)]$" "\$H[h(Y, Y)]$" "\$H[h(X, X)]$"
