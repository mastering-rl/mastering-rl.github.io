from model_free_reinforcement_learner import ModelFreeReinforcementLearner


class SARSA(ModelFreeReinforcementLearner):

    def state_value(self, state, action):
        return self.qfunction.get_q_value(state, action)

if __name__ == "__main__":
    from gridworld import CliffWorld
    from gridworld import GridWorld
    from qlearning import QLearning
    from qtable import QTable
    from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

    mdp = GridWorld()
    qfunction = QTable()
    SARSA(mdp, EpsilonGreedy(), qfunction).execute()
    policy = qfunction.extract_policy(mdp)
    print(mdp.q_function_to_string(qfunction))
    print(mdp.policy_to_string(policy))
    sarsa_rewards = mdp.get_rewards()
    print(sarsa_rewards)

    print("==========\nQ-learning: Cliffworld\n==========")

    mdp = CliffWorld()
    qfunction = QTable()
    QLearning(mdp, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)
    # Execute policy (using epsilon greedy with epsilon = 0.0
    QLearning(mdp, EpsilonGreedy(epsilon=0.0), qfunction=qfunction).execute(
        episodes=2000
    )
    q_learning_rewards = mdp.get_rewards()

    print("=====\nSARSA: Cliffworld\n=====")
    mdp = CliffWorld()
    qfunction = QTable()
    SARSA(mdp, EpsilonGreedy(epsilon=0.2), qfunction).execute(episodes=2000)
    # Execute policy (using epsilon greedy with epsilon = 0.0
    SARSA(mdp, EpsilonGreedy(epsilon=0.0), qfunction=qfunction).execute(episodes=2000)
    sarsa_rewards = mdp.get_rewards()

    from plot import Plot

    Plot.plot_rewards_per_episode(
        ["Q-learning", "SARSA"], [q_learning_rewards, sarsa_rewards]
    )
