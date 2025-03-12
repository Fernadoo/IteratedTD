from utils import TD_payoff, poisson

import numpy as np


class PoissonCH(object):
    """PoissonCH agent:
        1. Observe to update Gamma conjugate
           -> expected lambda for Poisson modelling
        2. BR to the Possion modelling
    """

    def __init__(self, name, gamma_a=15, gamma_b=10):
        self.name = name
        self.a = gamma_a
        self.b = gamma_b
        self.oppo_hist = []
        self.last_hi = 100

    def act(self, state):
        lo, hi = state.lo, state.hi
        if self.name == 'p1':
            oppo_hist = state.hist_a2
        elif self.name == 'p2':
            oppo_hist = state.hist_a1
        if len(oppo_hist) > 0:  # except for the first round
            self.oppo_hist.append(self.last_hi - oppo_hist[-1])
        self.last_hi = hi

        lam = (self.a + sum(self.oppo_hist)) / (self.b + len(self.oppo_hist))
        belief = [poisson(k, lam) for k in range(0, hi - lo + 1)]
        joint_u_self = [[TD_payoff(a1, a2)[0]
                         for a2 in range(hi, lo - 1, -1)]
                        for a1 in range(hi, lo - 1, -1)]
        exp_u1 = np.matmul(joint_u_self, belief)
        action = hi - np.argmax(exp_u1)

        return action
