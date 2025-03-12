import numpy as np

from utils import TD_payoff


class TitForTat(object):
    """Tit-For-Tat-in-spirit"""

    def __init__(self, name):
        self.name = name

    def act(self, state):

        lo, hi = state[:2]
        if self.name == 'p1':
            oppo_hist = state[3]
            # np.random.seed(0)
        elif self.name == 'p2':
            oppo_hist = state[2]
            # np.random.seed(1)

        if len(oppo_hist) == 0:
            return np.random.randint(lo, hi + 1)
        else:
            return oppo_hist[-1]


class PerfectlyRational(object):
    """Perfectly rational by playing NE from the beginning"""

    def __init__(self, name):
        self.name = name

    def act(self, state):
        lo, hi = state[:2]
        return lo


class Memory1BR(object):
    """Best response by bidding 1 unit less than the previous oppo_bid"""

    def __init__(self, name):
        self.name = name

    def act(self, state):

        lo, hi = state[:2]
        if self.name == 'p1':
            oppo_hist = state[3]
            # np.random.seed(0)
        elif self.name == 'p2':
            oppo_hist = state[2]
            # np.random.seed(1)

        if len(oppo_hist) == 0:
            return np.random.randint(lo, hi + 1)

        br = oppo_hist[-1] - 1
        return max(br, lo)


class HistoryBR(object):
    """Best response by according to the historical frequencies of oppo_bids
       => actually ficticious play
    """

    def __init__(self, name, beta=1.5):
        self.name = name
        self.beta = beta
        self.oppo_play_freq = np.zeros(99, dtype=int)
        self.last_hi = 100

    def act(self, state):

        lo, hi = state[:2]
        if self.name == 'p1':
            oppo_hist = state[3]
            u_idx = 0
        elif self.name == 'p2':
            oppo_hist = state[2]
            u_idx = 1

        if len(oppo_hist) > 0:
            oppo_last_a = self.last_hi - oppo_hist[-1]
            self.last_hi = hi
            self.oppo_play_freq[oppo_last_a] += 1
        exp_norm_freq = np.exp((self.oppo_play_freq - np.max(self.oppo_play_freq)) / self.beta)
        oppo_mixed_strategy = exp_norm_freq / np.sum(exp_norm_freq)
        joint_u_self = [[TD_payoff(a1, a2)[0]
                         for a2 in range(hi, lo - 1, -1)]
                        for a1 in range(hi, lo - 1, -1)]
        exp_u_self = np.matmul(joint_u_self, oppo_mixed_strategy)
        action = hi - np.argmax(exp_u_self)
        return action
