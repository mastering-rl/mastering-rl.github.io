from tabular_policy import TabularPolicy
from tabular_value_function import TabularValueFunction

class PolicyIteration:
    def __init__(self, mdp):
        self.mdp = mdp

    """ Calculate Q(s,a) of an action """

    def q_value(self, state, action, values):
        new_value = 0.0
        for (new_state, probability) in self.mdp.get_transitions(state, action):
            reward = self.mdp.get_reward(state, action, new_state)
            new_value += probability * (
                reward + (self.mdp.get_discount_factor() * values[new_state])
            )
        return new_value

    def policy_evaluation(self, policy, values, theta=0.001):

        while True:
            delta = 0.0
            new_values = dict()
            for state in self.mdp.get_states():
                # Calculate the value of V(s)
                old_value = values[state]
                values.update(state, self.q_value(state, policy.select_action(state), values))
                delta = max(delta, abs(old_value - values[state]))

            # terminate if the value function has converged
            if delta < theta:
                break

        return values

    """ Implmentation of policy iteration iteration """

    def policy_iteration(self, iterations=100, theta=0.001):

        # Initialise the value function V and policy function pi
        values = TabularValueFunction()
        policy = TabularPolicy(default_action=self.mdp.get_actions()[0])

        for i in range(iterations):
            policy_changed = False
            values = self.policy_evaluation(policy, values, theta)
            for state in self.mdp.get_states():
                old_action = policy.select_action(state)

                q_values = dict()
                for action in self.mdp.get_actions(state):
                    # Calculate the value of Q(s,a)
                    new_value = self.q_value(state, action, values)
                    q_values.update({action: new_value})

                # V(s) = argmax_a Q(s,a)
                new_action = max(q_values, key=lambda i: q_values[i])
                policy.update(state, new_action)

                policy_changed = (
                    True if new_action is not old_action else policy_changed
                )

            if not policy_changed:
                print("iterations = %d" % i)
                break

        return policy

    def initialise_value_function(self):
        return defaultdict(lambda: 0.0)


if __name__ == "__main__":
    from gridworld import GridWorld

    mdp = GridWorld(width=8, height=6)
    policy_iteration = PolicyIteration(mdp)

    for iterations in [0, 1, 2, 3, 4, 5, 10, 100]:
        print("After iteration " + str(iterations))
        policy = policy_iteration.policy_iteration(iterations=iterations).policy_table
        print(mdp.policy_to_string(policy) + "\n")
        mdp.visualise_policy(policy, title=f"num iterations={iterations}")
    mdp = GridWorld(width=20, height=15)
    policy_iteration = PolicyIteration(mdp)
    policy = policy_iteration.policy_iteration(iterations=100).policy_table
    print(mdp.policy_to_string(policy) + "\n")
    mdp.visualise_policy(policy, title=f"num iterations=100")
