from model_free_reinforcement_learner import ModelFreeReinforcementLearner


class QLearning(ModelFreeReinforcementLearner): 

    def state_value(self, state, action):
        (_, max_q_value) = self.qfunction.get_max_q(state, self.mdp.get_actions(state))
        return max_q_value

if __name__ == "__main__":
    from gridworld import *
    from qtable import QTable
    from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

    print("==========\nTabular Q-learning: Gridworld\n==========")
    mdp = GridWorld()
    qfunction = QTable()
    QLearning(mdp, EpsilonGreedy(), qfunction).execute()
    policy = qfunction.extract_policy(mdp)
    print(mdp.q_function_to_string(qfunction))
    print(mdp.policy_to_string(policy))
    q_learning_rewards = mdp.get_rewards()
    print(q_learning_rewards)

    print("==========\nLinear Q-Learning: Gridworld one terminal state\n==========")
    from linear_qfunction import LinearQFunction
    from gridworld_feature_extractor import GridWorldFeatureExtractor

    mdp = GridWorld(goals=[((3, 2), 1)])
    features = GridWorldFeatureExtractor(mdp)
    qfunction = LinearQFunction(features)
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes=2000)
    policy = qfunction.extract_policy(mdp)
    print(mdp.q_function_to_string(qfunction))
    print(mdp.policy_to_string(policy))


