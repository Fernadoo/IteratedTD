from collections import namedtuple
from tqdm import tqdm

import numpy as np

from utils import TD_payoff, TD_regret


State = namedtuple('State', ['lo', 'hi', 'hist_a1', 'hist_a2'])


class Game(object):
    """
    Iterative TD game:
    1. Upon same reported values, the game terminates
    2. Otherwise,
        i) self-transit to the same the current stage w.p. 0.5
        ii) transit to a shrunk stage with high := min(a1, a2)
    """

    def __init__(self, p1, p2, seed=0, max_len=100, trans_p=2):
        self.reset()
        # we set p1 as our AI agent and p2 as a pre-defined one
        self.p1 = p1
        self.p2 = p2
        self.hist_u1 = []
        self.hist_u2 = []
        self.hist_a1 = []
        self.hist_a2 = []
        self.hist_reg1 = []
        self.hist_reg2 = []

        # for iterative settings
        self.seed = seed
        self.cnt = 0
        self.max_len = max_len
        self.trans_p = trans_p

    def reset(self, lo=2, hi=100):
        self.lo = lo
        self.hi = hi

    def transit(self, a1, a2):
        u1, u2 = TD_payoff(a1, a2)
        self.hist_u1.append(u1)
        self.hist_u2.append(u2)
        self.hist_a1.append(a1)
        self.hist_a2.append(a2)

        reg1, reg2 = TD_regret(self.lo, self.hi, a1, a2)
        self.hist_reg1.append(reg1)
        self.hist_reg2.append(reg2)

        p = np.random.rand()
        if p >= self.trans_p:
            self.reset(self.lo, self.hi)
        else:
            # current ad-hoc transition rule
            self.reset(hi=max(a1, a2))

    def play(self):
        state = State(self.lo, self.hi, self.hist_a1, self.hist_a2)
        for _ in tqdm(range(self.max_len)):
            a1 = self.p1.act(state)
            a2 = self.p2.act(state)
            self.transit(a1, a2)
            self.cnt += 1

        return (self.hist_u1, self.hist_u2,
                self.hist_a1, self.hist_a2,
                self.hist_reg1, self.hist_reg2)


if __name__ == '__main__':
    from baselines import TitForTat, PerfectlyRational
    from ch_agents import PoissonCH

    # p1 = TitForTat('p1')
    p1 = PoissonCH('p1')
    # p2 = TitForTat('p2')
    p2 = PerfectlyRational('p2')

    ITDgame = Game(p1, p2, seed=0, epi_len_max=300)
    hist_u1, hist_u2, hist_a1, hist_a2, hist_reg1, hist_reg2 = ITDgame.play()
    print(u1_hist, u2_hist)
