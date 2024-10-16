from python_code.feature_extractors.gridworld_feature_extractor import (
    GridWorldFeatureExtractor,
)
from python_code.learners.qlearning import QLearning
from python_code.markov_decision_processes.gridworld import GridWorld
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.policies.q_policy import QPolicy
from python_code.qfunctions.linear_qfunction import LinearQFunction

mdp = GridWorld()
features = GridWorldFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
policy = QPolicy(qfunction)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))


from python_code.feature_extractors.gridworld_better_feature_extractor import (
    GridWorldBetterFeatureExtractor,
)

mdp = GridWorld()
features = GridWorldBetterFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
policy = QPolicy(qfunction)
print(mdp.q_function_to_string(qfunction))
print(mdp.policy_to_string(policy))
