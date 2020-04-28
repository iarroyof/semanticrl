RESULTS=/almac/ignacio/results_srl_env/wsize-8
AGENT="rdn"

for d in `ls $RESULTS`;
do
    ls --color=auto $RESULTS/${d}/${AGENT}_* | \
    parallel -k \
        'python semantic_reward.py --cols "\$H[h(Z, Z)]$" "\$H[h(Y, Y)]$" "\$H[h(X, X)]$" --in_csv {}' \
    > rs_${d}.txt; ls --color=auto $RESULTS/${d}/${AGENT}_* > files_${d}.txt;
    paste files_${d}.txt rs_${d}.txt > $RESULTS/${d}/${AGENT}_rewards.csv;
    sort_rewards.py $RESULTS/${d}/${AGENT}_rewards.csv;
done
