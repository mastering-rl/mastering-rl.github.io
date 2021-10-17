from gridworld import GridWorld
from qtable import QTable
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

print("==========\nTabular Q-learning: Gridworld\n==========")

mdp = GridWorld()
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=1000)
print(mdp.q_function_to_string(qfunction))

policy = qfunction.extract_policy(mdp)
print(mdp.policy_to_string(policy))

print('[', end="")
for i in range(mdp.width):
    print("[", end="")
    for j in range(mdp.height):
        print("(%d, %d), " % (i,j), end="")
        print( "%.2f, " % qfunction.get_q_value((i,j), mdp.UP), end="")
        print( "%.2f, " % qfunction.get_q_value((i,j), mdp.DOWN), end="")
        print( "%.2f, " % qfunction.get_q_value((i,j), mdp.RIGHT), end="")
        print( "%.2f]" % qfunction.get_q_value((i,j), mdp.LEFT))
print("]")

