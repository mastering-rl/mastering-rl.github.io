from model_free_reinforcement_learner import ModelFreeReinforcementLearner

class QLearning(ModelFreeReinforcementLearner):
    def update(self, state, action, next_state, next_action, reward):
        q_value = self.qfunction.get_q_value(state, action)
        (_, max_q_value) = self.qfunction.get_max_q(
            next_state, self.mdp.get_actions(next_state)
        )
        delta = self.alpha * (reward + self.mdp.discount_factor * max_q_value - q_value)
        return delta

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
    mdp = GridWorld(goals=[((3,2),1)])
    features = GridWorldFeatureExtractor(mdp)
    qfunction = LinearQFunction(features)
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes = 2000)
    policy = qfunction.extract_policy(mdp)
    print(mdp.q_function_to_string(qfunction))
    print(mdp.policy_to_string(policy))
    
    """
      
    print("=====\nSARSA: Gridworld\n=====")
    mdp = GridWorld(discount_factor = 0.9, width = 4, height = 3)
    qfunction = QTable()
    SARSA(mdp, EpsilonGreedy(), qfunction).execute(episodes = 1000)
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.qFunctionToString(qfunction))
    print(mdp.policyToString(policy))

    print("==========\nQ-learning: Cliffworld\n==========")
    
    mdp = CliffWorld()
    qfunction = QTable()
    QLearning(mdp, EpsilonGreedy(epsilon = 0.2), qfunction).execute(episodes = 2000)
    print(mdp.qFunctionToString(qfunction))
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.policyToString(policy))
    # Execute policy (using epsilon greedy with epsilon = 0.0
    QLearning(mdp, EpsilonGreedy(epsilon = 0.0), qfunction = qfunction).execute(episodes = 2000)
    qLearningRewards = mdp.getRewards()

    print("=====\nSARSA: Cliffworld\n=====")    
    mdp = CliffWorld()
    qfunction = QTable()
    SARSA(mdp, EpsilonGreedy(epsilon = 0.2), qfunction).execute(episodes = 2000)
    print(mdp.qFunctionToString(qfunction))
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.policyToString(policy))
    #mdp.visualiseQFunction(qFunction, title="SARSA: Cliffworld", showText=True)
    # Execute policy (using epsilon greedy with epsilon = 0.0
    SARSA(mdp, EpsilonGreedy(epsilon = 0.0), qfunction = qfunction).execute(episodes = 2000)
    sarsaRewards = mdp.getRewards()

    from plot import Plot
    Plot.plotRewardsPerEpisode(["Q-learning", "SARSA"], [qLearningRewards, sarsaRewards])


    """

