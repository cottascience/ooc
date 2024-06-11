import numpy as np
import math
import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
def num_tokens_from_string(string: str) -> int:
    num_tokens = len(encoding.encode(string))
    return num_tokens


def CI_score( y_hat, z, s, mode="diff"):
    assert mode in ["diff", "ratio"]
    s = [ tuple(ss) for ss in s ]
    score = []
    _y_hat = np.array(y_hat)
    _z = np.array(z)
    for adj in set(s):
        print(adj)
        idxs = [ i for i in range(len(s)) if s[i] == adj ]
        _zz = _z[idxs]
        _yy_hat = _y_hat[idxs]
        pz1 = _yy_hat[ [ i for i in range(len(_zz)) if _zz[i] == 1 ] ].mean()
        pz0 = _yy_hat[ [ i for i in range(len(_zz)) if _zz[i] == 0 ] ].mean()
        if (not math.isnan(pz1)) and (not math.isnan(pz0)):
            if mode == "diff":
                score += [ abs(pz1 - pz0) ]
            else:
                score += [ min([ pz1, pz0 ])/max([ pz1, pz0 ]) ]
    return np.array(score)