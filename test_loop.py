from srl_vs_rdn import SrlEnvTest
import random

NACTIONS = 28561  # Number of lines in openIE input file.

def rdn_range(a, b, N=10):
    """
    This method generates tuples of random pairs for n-gram ranges.
    The parameters must meet

        N < (((a - b) ** 2) - (a - b)) / 2

    where a < b are the interval's start and end, respectively. N is
    the desired number of pair tuples of integers.
    """
    possible = []
    n = (a - b)
    n = ((n ** 2) - n) / 2
    if n < N:
        print(
            "ERROR: N = {}, and (((a - b) ** 2) - (a - b)) / 2 = {}" \
                .format(N, n))
        return "ERROR"
    for x in range(a, b):
        for y in range(a, b):
            xy = tuple(sorted((x, y)))
            if x != y and not xy in possible:
                possible.append(xy)

    for tup in random.sample(possible, N):
        yield tup


def test_settings(n_tests):
    """
    bias=3, hitmiss=0.3, bw=5.0, density='expset', ngrams=(1, 3)
    """
    

def main():
    in_text = "data/dis_train_.txt"
    in_open = "data/dis_train.txt.oie"
    rdn_win = 10
    sampran = 500
    n_steps = 30
    samples = int(sampran * 0.3) 
    
    
    
    for s in random.sample(range(10, sampran), samples):
        nsteps = int(NACTIONS/s)
        out_dir = "results_" + str(s) + "_samples"
        test = SrlEnvTest(in_oie=in_open, in_txt=in_text, output_dir=out_dir,
                        wsize=rdn_win, sample=s, nsteps=nsteps)
        for param in settings:
            test.fit(bias=3, hitmiss=0.3, bw=5.0, density='expset', ngrams=(1, 3),
                                                                        output=None)
            
                    
