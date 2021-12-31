from gridworld import GridWorld
from qtable import QTable
from qlearning import QLearning
from linear_qfunction import LinearQFunction
from gridworld_better_feature_extractor import GridWorldBetterFeatureExtractor
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from side_by_side_comparison import run_learner
from side_by_side_comparison import join_gif

episodes=20
gridworld = GridWorld()
qfunction = QTable()
learner = QLearning(gridworld, EpsilonGreedy(), qfunction)
run_learner(mdp=gridworld, learner=learner, qfunction=qfunction, learner_name="Q-learning with Q-table", out_filename="../../assets/gifs/qlearning_with_qtable.gif", episodes=episodes)

gridworld = GridWorld()
features = GridWorldBetterFeatureExtractor(gridworld)
qfunction = LinearQFunction(features)
learner = QLearning(gridworld, EpsilonGreedy(), qfunction)
run_learner(mdp=gridworld, learner=learner, qfunction=qfunction, learner_name="Q-learning with linear function approximation", out_filename="../../assets/gifs/qlearning_with_linear_function_approximation.gif", episodes=episodes)

join_gif(filename1="../../assets/gifs/qlearning_with_qtable.gif", filename2="../../assets/gifs/qlearning_with_linear_function_approximation.gif", out_filename="../../assets/gifs/linear_function_vs_qtable.gif")
