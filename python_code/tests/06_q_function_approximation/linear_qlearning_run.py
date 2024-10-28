from python_code.feature_extractors.gridworld_better_feature_extractor import (
    GridWorldBetterFeatureExtractor,
)
from python_code.feature_extractors.gridworld_feature_extractor import (
    GridWorldFeatureExtractor,
)
from python_code.learners.qlearning import QLearning
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.linear_qfunction import LinearQFunction
from python_code.qfunctions.qtable import QTable
from python_code.tests.plot import Plot

mdp = GridWorld()
features = GridWorldFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
QLearning(mdp, EpsilonGreedy(), qfunction).execute()
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)


mdp = GridWorld()
features = GridWorldBetterFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
linear_qlearning_rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute()
policy = QPolicy(qfunction)
mdp.visualise_q_function(qfunction)
mdp.visualise_policy(policy)


mdp = GridWorld()
qfunction = QTable()
tabular_qlearning_rewards = QLearning(mdp, EpsilonGreedy(), qfunction).execute()


Plot.plot_cumulative_rewards(
    ["Tabular Q-learning", "Linear Q-learning"],
    [tabular_qlearning_rewards, linear_qlearning_rewards],
)
