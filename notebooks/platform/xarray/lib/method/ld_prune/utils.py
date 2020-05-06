

def dotvec1d(v1, v2):
    r = 0.0
    for i in range(len(v1)):
        r += v1[i] * v2[i]
    return r


def eqvec1d(v1, v2):
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            return 0
    return 1

