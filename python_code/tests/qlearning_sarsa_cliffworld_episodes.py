from gridworld import CliffWorld
from qtable import QTable
from qlearning import QLearning
from sarsa import SARSA
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

def train_and_test_cl(episodes):
# Train using Q-learning
    qfunction = QTable()
    cw = CliffWorld()
    QLearning(cw, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=episodes)
    # Extract the policy
    policy = qfunction.extract_policy(cw)
    # Execute the policy and get all rewards
    cw.execute_policy(policy, episodes=episodes)
    q_learning_rewards = cw.get_rewards()

    # Train using SARSA
    qfunction = QTable()
    cw = CliffWorld()
    SARSA(cw, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=episodes)
    # Execute the policy
    policy = qfunction.extract_policy(cw)
    cw.execute_policy(policy, episodes=episodes)
    sarsa_rewards = cw.get_rewards()
    return q_learning_rewards,sarsa_rewards


epcount=500
qvals=[]
svals=[]
for _ in range(5):
    q,s=train_and_test_cl(epcount)
    qvals.append(q)
    svals.append(s)

Plot.plot_multirun_rewards_per_episode(qvals,"Q-learning CliffWorld")
Plot.plot_multirun_rewards_per_episode(svals,"SARSA CliffWorld")

