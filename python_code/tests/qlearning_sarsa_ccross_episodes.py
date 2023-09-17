import contested_crossing
from qtable import QTable
from qlearning import QLearning
from sarsa import SARSA
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy
from tests.plot import Plot

def train_and_test_c(episodes):
# Train using Q-learning
    qfunction = QTable()
    ccross = contested_crossing.ContestedCrossing(high_danger=hd,low_danger=ld,ship_health=sh)
    QLearning(ccross, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=episodes)
    # Extract the policy
    policy = qfunction.extract_policy(ccross)
    # Execute the policy and get all rewards
    ccross.execute_policy(policy, episodes=episodes)
    q_learning_rewards = ccross.get_rewards()

    # Train using SARSA
    qfunction = QTable()
    ccross = contested_crossing.ContestedCrossing(high_danger=hd,low_danger=ld,ship_health=sh)
    SARSA(ccross, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=episodes)
    # Execute the policy
    policy = qfunction.extract_policy(ccross)
    ccross.execute_policy(policy, episodes=episodes)
    sarsa_rewards = ccross.get_rewards()
    return q_learning_rewards,sarsa_rewards


epcount=2000
qvals=[]
svals=[]
for _ in range(5):
    q,s=train_and_test_cl(epcount)
    qvals.append(q)
    svals.append(s)

Plot.plot_multirun_rewards_per_episode(qvals,"Q-learning Crossing")
Plot.plot_multirun_rewards_per_episode(svals,"SARSA Crossing")

epcount=20000
qvals=[]
svals=[]
for _ in range(5):
    q,s=train_and_test_cl(epcount)
    qvals.append(q)
    svals.append(s)

Plot.plot_multirun_rewards_per_episode(qvals,"Q-learning Crossing")
Plot.plot_multirun_rewards_per_episode(svals,"SARSA Crossing")
