from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from multi_armed_bandit.epsilon_decreasing import EpsilonDecreasing
from multi_armed_bandit.softmax import Softmax
from multi_armed_bandit.ucb import UpperConfidenceBounds

def plot_epsilon_greedy(drift=False):
    epsilon000 = EpsilonGreedy(epsilon=0.00).run_bandit(drift=drift)
    epsilon005 = EpsilonGreedy(epsilon=0.05).run_bandit(drift=drift)
    epsilon01 = EpsilonGreedy(epsilon=0.1).run_bandit(drift=drift)
    epsilon02 = EpsilonGreedy(epsilon=0.2).run_bandit(drift=drift)
    epsilon04 = EpsilonGreedy(epsilon=0.4).run_bandit(drift=drift)
    epsilon08 = EpsilonGreedy(epsilon=0.8).run_bandit(drift=drift)
    epsilon10 = EpsilonGreedy(epsilon=1.0).run_bandit(drift=drift)

    Plot.plot_rewards(
        [
            "epsilon = 0.0",
            "epsilon = 0.05",
            "epsilon = 0.1",
            "epsilon = 0.2",
            "epsilon = 0.4",
            "epsilon = 0.8",
            "epsilon = 1.0",
        ],
        [epsilon000, epsilon005, epsilon01, epsilon02, epsilon04, epsilon08, epsilon10],
    )


def plot_epsilon_decreasing(drift=False):
    alpha09 = EpsilonDecreasing(alpha=0.9).run_bandit(drift=drift)
    alpha099 = EpsilonDecreasing(alpha=0.99).run_bandit(drift=drift)
    alpha0999 = EpsilonDecreasing(alpha=0.999).run_bandit(drift=drift)
    alpha1 = EpsilonDecreasing(alpha=1.0).run_bandit(drift=drift)

    Plot.plot_rewards(
        ["alpha = 0.9", "alpha = 0.99", "alpha= 0.999", "alpha = 1.0"],
        [alpha09, alpha099, alpha0999, alpha1],
    )


def plot_softmax(drift=False):
    tau10 = Softmax(tau=1.0).run_bandit(drift=drift)
    tau11 = Softmax(tau=1.1).run_bandit(drift=drift)
    tau15 = Softmax(tau=1.5).run_bandit(drift=drift)
    tau20 = Softmax(tau=2.0).run_bandit(drift=drift)

    Plot.plot_rewards(
        ["tau = 1.0", "tau = 1.1", "tau = 1.5", "tau = 2.0"],
        [tau10, tau11, tau15, tau20],
    )


def plot_comparison(drift=False):
    epsilon_greedy = EpsilonGreedy(epsilon=0.1).run_bandit(drift=drift)
    epsilon_decreasing = EpsilonDecreasing(alpha=0.99).run_bandit(drift=drift)
    softmax = Softmax(tau=1.0).run_bandit(drift=drift)
    ucb = UpperConfidenceBounds().run_bandit(drift=drift)

    Plot.plot_rewards(
        [
            "Epsilon Greedy (epsilon = 0.1)",
            "Epsilon Decreasing (alpha = 0.99)",
            "Softmax (tau = 1.0)",
            "UCB",
        ],
        [epsilon_greedy, epsilon_decreasing, softmax, ucb],
    )


from plot import Plot

plot_epsilon_greedy()
plot_epsilon_decreasing()
plot_softmax(drift=False)
plot_softmax(drift=True)
plot_comparison(drift=False)
plot_comparison(drift=True)

