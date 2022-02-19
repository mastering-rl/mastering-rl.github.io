from actor_critic import ActorCritic
from qlearning import QLearning


class QActorCritic(ActorCritic):
    """
    This implements the actor critic algorithm using the Q-values from a Q-learning critic as the baseline.
    """

    def __init__(self, mdp, actor, critic, alpha=0.1):
        super().__init__(mdp, actor, critic, alpha)

    def update_actor(self, states, actions):
        q_values = [self.critic.qfunction.get_q_value(state, action) for state, action in zip(states, actions)]
        self.actor.policy.update(states, actions, q_values)

    def update_critic(self, reward, state, action, next_state):
        actions = self.mdp.get_actions(next_state)
        next_action = self.critic.bandit.select(next_state, actions, self.critic.qfunction)
        q_value = self.critic.qfunction.get_q_value(state, action)
        delta = self.critic.get_delta(reward, q_value, state, next_state, next_action)
        self.critic.qfunction.update(state=state, action=action, delta=delta)
        return delta
