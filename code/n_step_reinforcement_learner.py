class NStepReinforcementLearner:
    def __init__(self, mdp, bandit, qfunction, n, alpha=0.1):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.qfunction = qfunction
        self.n = n

    def execute(self, episodes=100):

        for i in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            last_n_rewards = []
            last_n_states = [state]
            last_n_actions = [action]

            while len(last_n_states) > 0:
                if not self.mdp.is_terminal(state):
                    (next_state, reward) = self.mdp.execute(state, action)
                    last_n_rewards += [reward]
                    actions = self.mdp.get_actions(next_state)

                    if not self.mdp.is_terminal(next_state):
                        next_action = self.bandit.select(
                            next_state, actions, self.qfunction
                        )

                if len(last_n_rewards) == self.n or self.mdp.is_terminal(state):
                    n_step_rewards = sum(
                        [
                            self.mdp.discount_factor ** (i + 1) * last_n_rewards[i]
                            for i in range(len(last_n_rewards))
                        ]
                    )
                    if len(last_n_rewards) == self.n:
                        next_state_value = self.state_value(next_state, next_action)
                        n_step_rewards = (
                            n_step_rewards
                            + self.mdp.discount_factor ** self.n * next_state_value
                        )

                    q_value = self.qfunction.get_q_value(
                        last_n_states[0], last_n_actions[0]
                    )
                    if i == 100:

                        print(
                            "update %s %s = %f"
                            % (
                                last_n_states[0],
                                last_n_actions[0],
                                (n_step_rewards - q_value),
                            )
                        )
                    self.qfunction.update(
                        last_n_states[0],
                        last_n_actions[0],
                        self.alpha * (n_step_rewards - q_value),
                    )

                    last_n_rewards = last_n_rewards[1 : self.n]
                    last_n_states = last_n_states[1 : self.n]
                    last_n_actions = last_n_actions[1 : self.n]

                if not self.mdp.is_terminal(state):
                    last_n_states += [state]
                    last_n_actions += [action]

                state = next_state
                action = next_action

    """ Get the value of a state """

    def state_value(self, state, action):
        abstract


class NStepQLearning(NStepReinforcementLearner):
    def state_value(self, state, action):
        (_, max_q_value) = self.qfunction.get_max_q(state, self.mdp.get_actions(state))
        return max_q_value


print("==========\nTabular n-step Q-learning: Gridworld\n==========")
from gridworld import GridWorld
from qtable import QTable
from qlearning import QLearning
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy


mdp = GridWorld(noise=0.0, goals=[((3, 2), 1)])
qfunction = QTable()
QLearning(mdp, EpsilonGreedy(), qfunction, alpha=0.4).execute(episodes=200)
# mdp.visualise_q_function(qfunction, "")
print(mdp.q_function_to_string(qfunction))

mdp = GridWorld(noise=0.0, goals=[((3, 2), 1)])
qfunction = QTable()
NStepQLearning(mdp, EpsilonGreedy(), qfunction, 5, alpha=0.4).execute(episodes=1)
mdp.visualise_q_function(qfunction, "")
# print(mdp.q_function_to_string(qfunction))

policy = qfunction.extract_policy(mdp)
mdp.visualise_policy(policy, "")
