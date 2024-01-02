import random
from temporal_difference_learner import TemporalDifferenceLearner

class ExperienceReplayLearner(TemporalDifferenceLearner):
    def __init__(self, mdp, bandit, qfunction, alpha=0.1, max_buffer_size=5000, replay_period=1000, batch_size=64):
        super().__init__(mdp, bandit, qfunction, alpha=alpha)
        self.buffer = []
        self.replay_period = replay_period
        self.batch_size = batch_size
        self.max_buffer_size = max_buffer_size

    def execute(self, episodes=100):

        for _ in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            t = 1
            while not self.mdp.is_terminal(state):
                (next_state, reward) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(next_state, actions, self.qfunction)
                q_value = self.qfunction.get_q_value(state, action)
                delta = self.get_delta(reward, q_value, state, next_state, next_action)

                experience = (state, action, delta)
                
                if (len(self.buffer) >= self.max_buffer_size):
                    self.buffer.pop(0)
                self.buffer.append(experience)

                if t % self.replay_period == 0:
                    self.update()

                state = next_state
                action = next_action

                t += 1

    """ Update from a mini batch """
    def update(self):
        mini_batch = random.sample(self.buffer, self.batch_size)
        self.qfunction.multi_update(mini_batch)

    def state_value(self, state, action):
        (_, max_q_value) = self.qfunction.get_max_q(state, self.mdp.get_actions(state))
        return max_q_value
