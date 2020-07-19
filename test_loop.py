from srl_vs_rdn import SrlEnvTest
import random
import numpy as np
import pandas as pd
import logging
from os import path, makedirs
logging.basicConfig(format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
from pdb import set_trace as st
# fit
NACTIONS = 28561  # Number of lines in openIE input file.
# dev
#NACTIONS = 18404

def rand_ranges(a, b, N):
    j = 0
    choice = list(range(a, b))
    ranges = []
    while j < N:
        rang = np.random.choice(choice, 2, p=[1./len(choice)] * len(choice))
        if len(set(rang)) > 1:
            ranges.append(tuple(sorted(rang)))
            j += 1

    return ranges


def test_settings(n_tests, ranges, steps=5):
    """
    Generate random parameters, e.g.:
        bias=3.0, hitmiss=0.3, bw=5.0,
        density='expset', ngrams={'low': 1, 'high': 5}
    """
    settings = {}
    for p, v in ranges.items():
        if not isinstance(v, dict):
            if not isinstance(v[0], str):
                if max(v) > 1:
                    values = np.linspace(v[0], v[1], steps)
                    settings[p] = np.random.choice(
                        [values[0]] + list(np.round(values[1:])),
                        n_tests, p=[1.0/steps] * steps)
                else:
                    settings[p] = np.random.choice(
                        np.linspace(v[0], v[1], steps),
                        n_tests, p=[1.0/steps] * steps)
            else:
                settings[p] = np.random.choice(
                    v, n_tests, p=[1.0/len(v)] * len(v)).tolist()
        else:
            settings[p] = rand_ranges(v['low'], v['high'], N=n_tests)

    sdf = pd.DataFrame(settings).drop_duplicates()
    sdf = sdf.reset_index(drop=True)
    last = sdf.index[-1]
    for i, s in sdf.iterrows():
        yield (i, last, dict(s.items()))

def main():
    in_text = "data/dis_train_.txt"
    in_open = "data/dis_train.txt.oie"
    out_dir = "/almac/ignacio/results_srl_env/"
#    in_text = "data/dis_test_.txt"
#    in_open = "data/dis_test.txt.oie"
#    out_dir = "/almac/ignacio/test_results_srl_env/"

    rdn_win = 8  # 6 + 2 (mean + std)
    sampran = (10, 320)  # sample range
    min_ngrams = 1
    max_ngrams = 5
    range_steps = 5
    samples = list(range(*sampran, 50))
    # (0. - 10.), (.1 - 1.), (1. - 5.), (expset, gausset), (1, 5)
    max_tests_possible =  (range_steps ** 3) * 2 * (2 ** max_ngrams)
    n_tests = int(max_tests_possible * 0.3)  # 25600 * 0.3 = 2560
    param_ranges = dict(
        bias=(0.0, 10.0),  # 10
        #hitmiss=(0.1, 1.0),  # 10
        bw=(0.000001, 0.01),  # 4
        #density=('expset', 'gausset'), # 2
        ngrams={'low': min_ngrams, 'high': max_ngrams} )

    settings = list(test_settings(n_tests=n_tests, steps=range_steps,
                                                        ranges=param_ranges))
    logging.info("Performing {} experiments in {} minutes (max.)" \
                   .format(len(samples) * n_tests, len(samples) * n_tests * 3))

    for s in samples:
        nsteps = int(float(NACTIONS)/float(s))
        out_dir = (
            path.normpath(out_dir)
                + "/wsize-" + str(rdn_win) + "/sample-" + str(s))
        if not path.exists(out_dir):
            makedirs(out_dir)
        
        test = SrlEnvTest(in_oie=in_open, in_txt=in_text, output_dir=out_dir,
                            wsize=rdn_win, sample=s, nsteps=nsteps)
        logging.info("Window and sample: {}".format(out_dir))
        for i, l, param in settings:
            logging.info(
                "Parameters test {}/{} ({:.1f} %) for {} samples: {}" \
                        .format(i, l, (i / l) * 100, s, param))
            try:
                test.fit(**param)
            except Exception as e:
                logging.error("ERROR: {}".format(e))


if __name__ == "__main__":
    main()
