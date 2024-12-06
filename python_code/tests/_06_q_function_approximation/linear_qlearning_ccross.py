from python_code.feature_extractors.ccross_feature_extractor import (
    CCrossFeatureExtractor,
)
from python_code.learners.qlearning import QLearning
from python_code.markov_decision_processes import contested_crossing
from python_code.multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from python_code.policies.stochastic_q_policy import StochasticQPolicy
from python_code.qfunctions.linear_qfunction import LinearQFunction

mdp = contested_crossing.ContestedCrossing()
features = CCrossFeatureExtractor(mdp)
qfunction = LinearQFunction(features)
QLearning(mdp, EpsilonGreedy(), qfunction).execute()
policy = StochasticQPolicy(qfunction)
mdp.visualise_as_image(
    policy=policy,
    mode=0,
    title="Low danger: {0}, High danger: {1}".format(mdp.low_danger, mdp.high_danger),
    plot=True,
)
