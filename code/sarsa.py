from model_free_reinforcement_learner import ModelFreeReinforcementLearner

class SARSA(ModelFreeReinforcementLearner):
    def update(self, state, action, next_state, next_action, reward):
        q_value = self.qfunction.get_q_value(state, action)
        q_value_next = self.qfunction.get_q_value(next_state, next_action)
        delta = self.alpha * (reward + self.mdp.discount_factor * q_value_next - q_value)
        return delta

if __name__ == "__main__":
    from gridworld import *
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
