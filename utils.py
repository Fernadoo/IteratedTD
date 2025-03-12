import numpy as np


def poisson(k, lam):
    return lam ** k * np.exp(-lam) / np.math.factorial(k)


def TD_payoff(a1, a2):
    return min(a1, a2) - 2 * np.sign(a1 - a2), min(a1, a2) + 2 * np.sign(a1 - a2)


def TD_regret(lo, hi, a1, a2):
    u1, u2 = TD_payoff(a1, a2)
    best_a1 = max(a2 - 1, lo)
    reg1 = TD_payoff(best_a1, a2)[0] - u1
    best_a2 = max(a1 - 1, lo)
    reg2 = TD_payoff(a1, best_a2)[1] - u2
    return reg1, reg2
