class ActorCritic:
    def __init__(self, mdp, actor, critic):
        self.mdp = mdp
        self.actor = actor  # Actor (policy based) to select actions
        self.critic = critic  # Critic (value based) to evaluate actions

    def execute(self, episodes=100):
        episode_rewards = []
        for episode in range(episodes):
            actions = []
            states = []
            rewards = []
            next_states = []

            episode_reward = 0
            step = 0
            state = self.mdp.get_initial_state()
            while not self.mdp.is_terminal(state):
                action = self.actor.select_action(state, self.mdp.get_actions(state))
                (next_state, reward, done) = self.mdp.execute(state, action)
                self.update_critic(reward, state, action, next_state)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)

                state = next_state

                episode_reward += reward * (self.mdp.discount_factor ** step)
                step += 1

            self.update_actor(
                rewards=rewards, states=states, actions=actions, next_states=next_states
            )

            episode_rewards.append(episode_reward)
        return episode_rewards

    """ Update the actor using a batch of rewards, states, actions, and next states """

    def update_actor(self, rewards, states, actions, next_states):
        abstract

    """ Update the critc using a reward, state, action, and next state """

    def update_critic(self, reward, state, action, next_state):
        abstract
