from actor_critic import ActorCritic


class QActorCritic(ActorCritic):
    def __init__(self, mdp, actor, critic):
        super().__init__(mdp, actor, critic)

    def update_actor(self, rewards, states, actions, next_states, dones):
        q_values = self.critic.get_q_values(states, actions)
        next_state_q_values = self.state_values(next_states, actions)
        deltas = [
            reward + (self.mdp.get_discount_factor() * next_state_q_value) - q_value
            if not done else (reward - q_value)
            for reward, next_state_q_value, q_value, done in zip(
                rewards, next_state_q_values, q_values, dones
            )
        ]

        self.actor.update(states, actions, deltas)

    def update_critic(self, reward, state, action, next_state):
        state_value = self.critic.get_q_value(state, action)
        actions = self.mdp.get_actions(next_state)
        next_state_value = self.critic.get_max_q(next_state, actions)
        delta = reward + self.mdp.get_discount_factor() * next_state_value - state_value
        self.critic.update(state, delta)

    '''
    def state_values(self, states, actions):
        return self.critic.q_get_values(states)

    def update_critic(self, reward, state, action, next_state):
        actions = self.mdp.get_actions(next_state)
        next_action = self.actor.select_action(next_state, actions)
        delta = self.get_delta(reward, state, action, next_state, next_action)
        self.critic.update(state, action, delta)

    def get_delta(self, reward, state, action, next_state, next_action):
        q_value = self.critic.get_q_value(state, action)
        next_state_value = self.state_value(next_state, next_action)
        delta = reward + self.mdp.get_discount_factor() * next_state_value - q_value
        return delta
    
    def state_values(self, states, actions):
        return self.critic.get_max_q_values(states)
    
    def state_value(self, state, action):
        return self.critic.get_max_q(state, self.mdp.get_actions(state))
    '''