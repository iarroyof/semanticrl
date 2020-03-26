import random

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
