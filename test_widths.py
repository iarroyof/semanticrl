from srl_vs_rdn import SrlEnvTest
from os import path
import numpy as np

in_text = "data/dis_test_.txt"
in_open = "data/dis_test.txt.oie"
out_dir = "/almac/ignacio/test_results_srl_env/"

rdn_win = 8
nsteps = 50
s = 260

out_dir_ = (
            path.normpath(out_dir)
                + "/wsize-" + str(rdn_win) + "/sample-" + str(s))


test = SrlEnvTest( 
    in_oie=in_open,
    in_txt=in_text,
    output_dir=out_dir_,
    wsize=rdn_win,
    sample=s,
    nsteps=nsteps,
    to_simulate='oie',
    return_df=True
)

widths = np.linspace(1.5, 2.5, 10, endpoint=True)
measure_type = 'h'

cols, mname = {
    'h':
        (["$H[h(Z, Z)]$", "$H[h(Y, Y)]$", "$H[h(X, X)]$"], "HX_HY_HZ"),
    'cmi':
        (["$I[h(Y+Z, X)]$", "$I[h(X+Z, Y)]$", "$I[h(X+Y, Z)]$"], "IXYZ_IYZX_IXZY"),
    'mi':
        (["$I[h(Y, X)]$", "$I[h(Z, X)]$", "$I[h(Z, Y)]$"], "IXY_IYZ_IXZ"),
    'jh':
        (["$H[h(Y+Z, Y+Z)]$", "$H[h(X+Z, X+Z)]$", "$H[h(X+Y, X+Y)]$"], "HXY_HYZ_HXZ")
}[measure_type]

for w in widths:
    param = {'bias': 1.0, 'bw': w, 'ngrams': (1, 4)}

    oie_df, _ = test.fit(**param)

    rewards = test.semantic_reward(
        cols=cols,
        measure=mname,
        sample=None,
        csv=None,
        in_df=oie_df,
        beta=1e3
    )
    rewards['width'] = w
    rewards['measure'] = mname
    print(rewards)
