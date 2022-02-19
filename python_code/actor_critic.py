class ActorCritic:
    def __init__(self, mdp, actor, critic, alpha=0.1):
        self.alpha = alpha  # Learning rate (gradient update step-size)
        self.mdp = mdp
        self.actor = actor  # Actor (policy based) to select actions
        self.critic = critic  # Critic (value based) to evaluate actions

    def execute(self, episodes=100):
        for _ in range(episodes):
            actions = []
            states = []
            rewards = []

            state = self.mdp.get_initial_state()
            while not self.mdp.is_terminal(state):
                action = self.actor.select_action(state)
                next_state, reward = self.mdp.execute(state, action)
                critic_deltas = self.calculate_critic_delta(reward, state, action, next_state)
                self.critic.update(states=states, actions=actions, deltas=critic_deltas)

                # Store the information from this step of the trajectory
                states.append(state)
                actions.append(action)
                rewards.append(reward)

                state = next_state

            actor_baselines = self.calculate_actor_baseline(states, actions)
            self.actor.update(states=states, actions=actions, deltas=actor_baselines)

    def calculate_actor_baseline(self, states, actions):
        abstract

    def calculate_critic_delta(self, reward, state, action, next_state):
        abstract



