import numpy as np

from utils import TD_payoff


class MemoryK(object):
    """Best response by according to the historical frequencies of oppo_bids
       => actually ficticious play
    """

    def __init__(self, name, k=1):
        self.name = name
        self.k = k
        self.oppo_play_freq = np.zeros(99, dtype=int)
        self.last_hi = 100

    def act(self, state):

        lo, hi = state[:2]
        if self.name == 'p1':
            oppo_hist = state[3]
        elif self.name == 'p2':
            oppo_hist = state[2]

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