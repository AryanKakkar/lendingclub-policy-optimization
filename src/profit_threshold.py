import numpy as np

def break_even_p(int_rate, lgd=1.0, cost=0.0):
    """
    Approve if (1-p)*G - p*L > 0 => p < G/(L+G)
    G ~ int_rate (per-loan interest rate), L ~ LGD (scaled by principal)
    """
    G = np.maximum(np.asarray(int_rate) - cost, 0.0)
    L = lgd
    return G / (L + G + 1e-12)

def decisions_from_probs(p_default, int_rate, lgd=1.0, cost=0.0):
    thr = break_even_p(int_rate, lgd=lgd, cost=cost)
    return (np.asarray(p_default) < thr).astype(int)  # 1=approve, 0=deny
