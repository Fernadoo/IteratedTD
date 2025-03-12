import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Process some options for the agents.')

    # Adding arguments
    parser.add_argument(
        '-a', '--agents',
        type=str,
        nargs=2,
        required=True,
        help='Types of two agents (e.g. --agents PoissonCH PerfectlyRational)'
    )

    parser.add_argument(
        '-l', '--max_len',
        type=int,
        default=100,
        help='Maximum length of iteration (default: 100)'
    )

    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=0,
        help='Random seed (default: 0)'
    )

    parser.add_argument(
        '-sp', '--show_payoff',
        type=int,
        choices=[0, 1, 2],
        help='Show payoff (0: No, 1: Yes but Numeric, 2: Numeric and plot)'
    )

    parser.add_argument(
        '-sw', '--show_win_rate',
        type=int,
        choices=[0, 1, 2],
        help='Show win rate (0: No, 1: Yes but Numeric, 2: Numeric and plot)'
    )

    parser.add_argument(
        '-sr', '--show_regret',
        type=int,
        choices=[0, 1, 2],
        help='Show regret (0: No, 1: Yes but Numeric, 2: Numeric and plot)'
    )

    # Parse the arguments
    args = parser.parse_args()

    # Print the parsed arguments (for demonstration purposes)
    print(f'Types of agents: {args.agents}')
    print(f'Max length of iteration: {args.max_len}')
    print(f'Random seed: {args.seed}')
    print(f'Show payoff: {args.show_payoff}')
    print(f'Show win rate: {args.show_win_rate}')
    print(f'Show regret: {args.show_regret}')

    return args


if __name__ == '__main__':
    from baselines import TitForTat, PerfectlyRational, Memory1BR, HistoryBR
    from ch_agents import PoissonCH
    from env import Game

    import numpy as np

    print('--- Settings ---')
    args = get_args()

    p1 = locals()[args.agents[0]]('p1')
    p2 = locals()[args.agents[1]]('p2')

    print('\n--- Game Starts ---')
    ITDgame = Game(p1, p2, seed=args.seed, max_len=args.max_len)
    hist_u1, hist_u2, hist_a1, hist_a2, hist_reg1, hist_reg2 = ITDgame.play()

    if (not args.show_payoff) and (not args.show_win_rate) and not (args.show_regret):
        exit(0)
    fig = None
    if args.show_payoff == 2 or args.show_win_rate == 2 or args.show_regret == 2:
        from matplotlib import pyplot as plt
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), dpi=100)

    print('\n--- Result Summary ---')

    # total payoff
    if args.show_payoff >= 1:
        print(f"Total Payoff: P1 ({np.sum(hist_u1)}) v.s. P2 ({np.sum(hist_u2)})")
    if args.show_payoff == 2:
        axes[0][0].plot(np.arange(args.max_len), hist_a1, color='tab:blue', label='p1')
        axes[0][0].plot(np.arange(args.max_len), hist_a2, color='tab:orange', label='p2')
        axes[0][0].legend()
        axes[0][0].set_title('Payoff')

        steps = np.linspace(0, 1, args.max_len)
        alphas = (1 - steps) * 0.2 + steps * 1
        sizes = (1e3 - 10) * np.exp(-12 * steps) + 10
        # sizes = (1- steps) * 1e3 + (steps) * 10
        axes[0][1].scatter(hist_a1, hist_a2,
                           alpha=alphas, sizes=sizes, edgecolor='none', facecolor='tab:green')
        axes[0][1].text(x=hist_a1[-1], y=hist_a2[-1], s=f"{(hist_a1[-1], hist_a2[-1])}",
                        ha='center', va='center', color='tab:red')
        axes[0][1].set_xlabel('P1')
        axes[0][1].set_ylabel('P2')
        axes[0][1].set_title('Bidding profile')

    # total win rate
    if args.show_win_rate > 0:
        p1_win = np.array(hist_u1) > np.array(hist_u2)
        p2_win = np.array(hist_u1) < np.array(hist_u2)
        p1_win_total = np.sum(p1_win)
        p2_win_total = np.sum(p2_win)
        tie_total = args.max_len - p1_win_total - p2_win_total
        final_winner = ['p2', 'p1'][int(p1_win_total > p2_win_total)]
    if args.show_win_rate >= 1:
        print(f"Win rate: P1 ({p1_win_total}) v.s. P2 ({p2_win_total}), "
              f"tie ({tie_total}), in total: {final_winner}")
    if args.show_win_rate == 2:
        w = np.zeros(args.max_len, dtype=int)
        w[p1_win > 0] = 1
        w[p2_win > 0] = -1
        cum_win = np.cumsum(w)
        axes[1][0].plot(np.arange(args.max_len), np.zeros(args.max_len), color='grey', linewidth=1)
        axes[1][0].fill_between(np.arange(args.max_len), cum_win, where=(cum_win >= 0),
                                color='tab:blue', alpha=0.5, label='P1 Wins')
        axes[1][0].fill_between(np.arange(args.max_len), cum_win, where=(cum_win <= 0),
                                color='tab:orange', alpha=0.5, label='P2 Wins')
        axes[1][0].legend()
        axes[1][0].set_title('Cumulative win rate')

    # total regret
    if args.show_regret >= 1:
        print(f"Total Regret: P1 ({np.sum(hist_reg1)}) v.s. P2 ({np.sum(hist_reg2)})")
    if args.show_regret == 2:
        axes[1][1].plot(np.arange(args.max_len), hist_reg1, color='tab:blue', label='p1')
        axes[1][1].plot(np.arange(args.max_len), hist_reg2, color='tab:orange', label='p2')
        axes[1][1].legend()
        axes[1][1].set_title('Regret')

    if fig:
        plt.tight_layout()
        plt.show()
