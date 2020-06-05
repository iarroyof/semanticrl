# Execute with:
# for m in {mi,cmi,h,jh}; do bash print_results.sh $m; done
RESULTS=/almac/ignacio/results_srl_env/wsize-8
AGENT="rdn"

for d in `ls $RESULTS`;
do
    { if [[ $1 = cmi ]]; then
        POSFIJO=IXYZ_IYXZ_IZXY
        ls $RESULTS/${d}/${AGENT}_bias-* | parallel -k 'python semantic_reward.py --cols "\$I[h(Y+Z, X)]$" "\$I[h(X+Z, Y)]$" "\$I[h(X+Y, Z)]$" --in_csv {}' > rs_${d}.txt
    fi;
    if [[ $1 = h ]]; then
        POSFIJO=HX_HY_HZ
        ls $RESULTS/${d}/${AGENT}_bias-* | parallel -k 'python semantic_reward.py --cols "\$H[h(Z, Z)]$" "\$H[h(Y, Y)]$" "\$H[h(X, X)]$" --in_csv {}' > rs_${d}.txt
    fi;
    if [[ $1 = mi ]]; then
        POSFIJO=IZY_IZX_IYX;
        ls $RESULTS/${d}/${AGENT}_bias-* | parallel -k 'python semantic_reward.py --cols "\$I[h(Y, X)]$" "\$I[h(Z, X)]$" "\$I[h(Z, Y)]$" --in_csv {}' > rs_${d}.txt
    fi;
    if [[ $1 = jh ]]; then
        POSFIJO=HXY_HXZ_HYZ
        ls $RESULTS/${d}/${AGENT}_bias-* | parallel -k 'python semantic_reward.py --cols "\$H[h(Y+Z, Y+Z)]$" "\$H[h(X+Z, X+Z)]$" "\$H[h(X+Y, X+Y)]$" --in_csv {}' > rs_${d}.txt
    fi; } && \
    ls $RESULTS/${d}/${AGENT}_bias-* > files_${d}.txt && \
    paste files_${d}.txt rs_${d}.txt > $RESULTS/${d}/${AGENT}_rewards_${POSFIJO}.csv && \
    python sort_rewards.py $RESULTS/${d}/${AGENT}_rewards_${POSFIJO}.csv;
    echo $d;
done
echo $POSFIJO
AGENT="oie"
echo $AGENT
for d in `ls $RESULTS`;
do  { if [[ $1 = cmi ]]; then
        POSFIJO=IXYZ_IYXZ_IZXY
        ls $RESULTS/${d}/${AGENT}_bias-* | parallel -k 'python semantic_reward.py --cols "\$I[h(Y+Z, X)]$" "\$I[h(X+Z, Y)]$" "\$I[h(X+Y, Z)]$" --in_csv {}' > rs_${d}.txt
    fi;
    if [[ $1 = h ]]; then
        POSFIJO=HX_HY_HZ
        ls $RESULTS/${d}/${AGENT}_bias-* | parallel -k 'python semantic_reward.py --cols "\$H[h(Z, Z)]$" "\$H[h(Y, Y)]$" "\$H[h(X, X)]$" --in_csv {}' > rs_${d}.txt
    fi;
    if [[ $1 = mi ]]; then
        POSFIJO=IZY_IZX_IYX;
        ls $RESULTS/${d}/${AGENT}_bias-* | parallel -k 'python semantic_reward.py --cols "\$I[h(Y, X)]$" "\$I[h(Z, X)]$" "\$I[h(Z, Y)]$" --in_csv {}' > rs_${d}.txt
    fi;
    if [[ $1 = jh ]]; then
        POSFIJO=HXY_HXZ_HYZ
        ls $RESULTS/${d}/${AGENT}_bias-* | parallel -k 'python semantic_reward.py --cols "\$H[h(Y+Z, Y+Z)]$" "\$H[h(X+Z, X+Z)]$" "\$H[h(X+Y, X+Y)]$" --in_csv {}' > rs_${d}.txt
    fi; } && \
    ls $RESULTS/${d}/${AGENT}_bias-* > files_${d}.txt && \
    paste files_${d}.txt rs_${d}.txt > $RESULTS/${d}/${AGENT}_rewards_${POSFIJO}.csv && \
    python sort_rewards.py $RESULTS/${d}/${AGENT}_rewards_${POSFIJO}.csv;
    echo $d;
done


## sorted_sample-110_rdn_rewards_HX_HY_HZ.csv
## "\$H[h(Z, Z)]$" "\$H[h(Y, Y)]$" "\$H[h(X, X)]$"
## "\$I[h(Y+Z, X)]$" "\$I[h(X+Z, Y)]$" "\$I[h(X+Y, Z)]$"
## "\$I[h(Y, X)]$" "\$I[h(Z, X)]$" "\$I[h(Z, Y)]$"
## "\$H[h(Y+Z, Y+Z)]$" "\$H[h(X+Z, X+Z)]$" "\$H[h(X+Y, X+Y)]$"
