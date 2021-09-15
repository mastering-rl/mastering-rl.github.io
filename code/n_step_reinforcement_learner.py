class NStepReinforcementLearner:
    def __init__(self, mdp, bandit, qfunction, n, alpha=0.4):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.qfunction = qfunction
        self.n = n

    def execute(self, episodes=100):

        for _ in range(episodes):
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
                        next_action = self.bandit.select(next_state, actions, self.qfunction)

                if len(last_n_rewards) == self.n:
                    delta = sum([self.mdp.discount_factor**i * (last_n_rewards[i]) for i in range(self.n)])
                    if len(last_n_rewards) == self.n:
                        q_value = self.qfunction.get_q_value(state, action)
                        next_state_value = self.state_value(next_state, next_action)
                        delta = reward + self.mdp.discount_factor**self.n * next_state_value - q_value
                        
                    self.qfunction.update(state, action, self.alpha * delta)
                    
                    last_n_rewards = last_n_rewards[1:]
                    last_n_states = last_n_states[1:]
                    last_n_actions = last_n_actions[1:]

                if not self.mdp.is_terminal(next_state):
                    last_n_states += [state]
                    last_n_actions += [action]
                else:
                    last_n_rewards = last_n_rewards[1:]
                    last_n_states = last_n_states[1:]
                    last_n_actions = last_n_actions[1:]

                state = next_state
                action = next_action
                print(last_n_states)


    """ Get the value of a state """

    def state_value(self, state, action):
        abstract


class NStepQLearning(NStepReinforcementLearner):
    def state_value(self, state, action):
        (_, max_q_value) = self.qfunction.get_max_q(state, self.mdp.get_actions(state))
        return max_q_value


print("==========\nTabular s-step Q-learning: Gridworld\n==========")
from gridworld import GridWorld
from qtable import QTable
from multi_armed_bandit.epsilon_greedy import EpsilonGreedy

mdp = GridWorld(goals=[((3,2), 1)])
qfunction = QTable()
NStepQLearning(mdp, EpsilonGreedy(), qfunction, 3).execute(episodes=2)
mdp.visualise_q_function(qfunction, "")

policy = qfunction.extract_policy(mdp)
#mdp.visualise_policy(policy, "")
