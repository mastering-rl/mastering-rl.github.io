from model_free_reinforcement_learner import ModelFreeReinforcementLearner

class RewardShapedQLearning(ModelFreeReinforcementLearner):
    def __init__(self, mdp, bandit, potential, qfunction, alpha=0.1):
        super().__init__(mdp, bandit, qfunction=qfunction, alpha=alpha)
        self.potential = potential

    def update(self, state, action, next_state, next_action, reward):
        (_, max_q_value) = self.qfunction.get_max_q(
            next_state, self.mdp.get_actions(next_state)
        )
        q_value = self.qfunction.get_q_value(state, action)
        state_potential = self.potential.get_potential(state)
        next_state_potential = self.potential.get_potential(next_state)
        potential = self.mdp.discount_factor * next_state_potential - state_potential
        delta = self.alpha * (
            reward + potential + self.mdp.discount_factor * max_q_value - q_value
        )
        return delta

if __name__ == "__main__":
    from gridworld import *
    from qtable import QTable
    from qlearning import QLearning
    from gridworld_potential_function import GridWorldPotentialFunction
    from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

    print("==========\nTabular Q-learning: Gridworld\n==========")
    mdp = GridWorld()
    qfunction = QTable()
    QLearning(mdp, EpsilonGreedy(), qfunction).execute()
    policy = qfunction.extract_policy(mdp)
    print(mdp.q_function_to_string(qfunction))
    print(mdp.policy_to_string(policy))
    q_learning_rewards = mdp.get_rewards()
    
    print("==========\nReward Shaped Q-learning: Gridworld\n==========")
    mdp = GridWorld()
    qfunction = QTable()
    potential = GridWorldPotentialFunction(mdp)
    RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute()
    policy = qfunction.extract_policy(mdp)
    print(mdp.q_function_to_string(qfunction))
    print(mdp.policy_to_string(policy))
    reward_shaped_rewards = mdp.get_rewards()

    from plot import Plot

    Plot.plot_episode_length(
        ["Tabular Q-learning", "Reward shaping"],
        [q_learning_rewards, reward_shaped_rewards],
    )
