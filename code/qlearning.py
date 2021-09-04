from qtable import QTable

class ModelFreeReinforcementLearner():

    def __init__(self, mdp, bandit, qfunction = QTable(), alpha = 0.1):
        self.mdp = mdp
        self.bandit = bandit
        self.alpha = alpha
        self.qfunction = qfunction

    def execute(self, episodes = 100):

        for i in range(episodes):
            state = self.mdp.get_initial_state()
            actions = self.mdp.get_actions(state)
            action = self.bandit.select(state, actions, self.qfunction)

            while not self.mdp.is_terminal(state):
                (next_state, reward) = self.mdp.execute(state, action)
                actions = self.mdp.get_actions(next_state)
                next_action = self.bandit.select(next_state, actions, self.qfunction)
                new_q_value = self.update(state, action, next_state, next_action, reward)
                self.qfunction.update(state, action, new_q_value)
                state = next_state
                action = next_action

    '''
        Update a Q-function with a new delta
    '''
    def update(self, state, action, next_state, reward): abstract


from multi_armed_bandits import EpsilonGreedy

class QLearning(ModelFreeReinforcementLearner):
    def update(self, state, action, next_state, next_action, reward):
        q_value = self.qfunction.get_q_value(state, action)
        (_, max_q_value) = self.qfunction.get_max_q(next_state, self.mdp.get_actions(next_state))
        delta = self.alpha * (reward + self.mdp.discount_factor * max_q_value - q_value)
        return delta
    
class SARSA(ModelFreeReinforcementLearner):
    def update(self, state, action, next_state, next_action, reward): 
        q_value = self.qfunction.get_q_value(state, action)
        q_value_next = self.qfunction.get_q_value(next_state, next_action)
        delta = self.alpha * (reward + self.mdp.discount_factor * q_value_next - q_value)
        return delta


class FeatureExtractor():
    def extract_features(self, state, action): abstract

class GridWorldStateFeatureExtractor(FeatureExtractor):
    def __init__(self, width, height):
        self.width = width
        self.height = height

class GridWorldFeatureExtractor(FeatureExtractor):
    def __init__(self, mdp):
        self.mdp = mdp

    def num_features(self):
        return 5

    def num_actions(self):
        return len(self.mdp.get_actions())
    
    def extract(self, state, action):
        goal = (self.mdp.width - 1, self.mdp.height - 1)
        x = 0
        y = 1
        e = 0.01
        feature_values = []
        for a in self.mdp.get_actions():
            if a == action and state != GridWorld.TERMINAL:
                #featureValues += [1 - ((goal[x] - state[x]) / goal[x]) + 0.01]
                #featureValues += [1 - ((goal[y] - state[y]) / goal[y]) + 0.01]
                #featureValues += [1 - ((goal[x] - state[x] + goal[y] - state[y]) / (goal[x] + goal[y])) + 0.01]
                feature_values += [(state[x] + e)/(goal[x] + e)]
                feature_values += [(state[y] + e)/(goal[y] + e)]
                feature_values += [(goal[x] - state[x] + goal[y] - state[y] + e) / (goal[x] + goal[y] + e)]
                feature_values += [1 if goal[x] == state[x] else 0]
                feature_values += [1 if goal[y] == state[y] else 0]
                #featureValues += [1 if 0 == state[x] else 0]
                #featureValues += [1 if goal[y] - 1 == state[y] else 0]
                #featureValues += [1 if state == (goal[x], goal[y]) else 0]
                #featureValues += [1 if state == (0, goal[y] - 1)else 0]
            else:
                for _ in range(0, self.num_features()):
                    feature_values += [0.0]
        return feature_values

class RewardShapedQLearning(QLearning):
    def __init__(self, mdp, bandit, potential, qfunction = QTable(), alpha = 0.1):
        super().__init__(mdp, bandit, qfunction = qfunction, alpha = alpha)
        self.potential = potential

    def update(self, state, action, next_state, next_action, reward): 
        (_, max_q_value) = self.qfunction.get_max_q(next_state, self.mdp.get_actions(next_state))
        q_value = self.qfunction.get_q_value(state, action)
        state_potential = self.potential.get_potential(state)
        next_state_potential = self.potential.get_potential(next_state)
        potential = self.mdp.discount_factor *  next_state_potential - state_potential
        delta = self.alpha * (reward + potential + self.mdp.discount_factor * max_q_value - q_value)
        return delta


class PotentialFunction():
    def get_potential(self, state): abstract

from gridworld import GridWorld

class GridWorldPotentialFunction(PotentialFunction):

    def __init__(self, mdp):
        self.mdp = mdp
        
    def get_potential(self, state):
        if state != GridWorld.TERMINAL:
            goal = (self.mdp.width, self.mdp.height)
            x = 0
            y = 1
            return 0.1 * (1 - ((goal[x] - state[x] + goal[y] - state[y]) / (goal[x] + goal[y])))
        else:
            return 0.0
        

if __name__ == "__main__":
    from gridworld import *
    
    print("==========\nTabular Q-learning: Gridworld\n==========")
    width = 10
    height = 10
    goals = [((width - 1, height - 1), 1)]
    episodes = 50
    mdp = GridWorld(width = width, height = height, goals = goals)
    print(mdp.visualise())
    #mdp = GridWorld(width = 15, height = 12, goals = [((14,11), 1), ((13,11), -1)])
    qfunction = QTable()
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes = episodes)
    policy = mdp.extract_policy_from_q_function(qfunction)
    print(mdp.q_function_to_string(qfunction))
    print(mdp.policy_to_string(policy))
    q_learning_rewards = mdp.get_rewards()
    print(q_learning_rewards)
    '''
      
    print("=====\nSARSA: Gridworld\n=====")
    mdp = GridWorld(discount_factor = 0.9, width = 4, height = 3)
    qfunction = QTable()
    SARSA(mdp, EpsilonGreedy(), qfunction).execute(episodes = 1000)
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.qFunctionToString(qfunction))
    print(mdp.policyToString(policy))

    print("==========\nQ-learning: Cliffworld\n==========")
    
    mdp = CliffWorld()
    qfunction = QTable()
    QLearning(mdp, EpsilonGreedy(epsilon = 0.2), qfunction).execute(episodes = 2000)
    print(mdp.qFunctionToString(qfunction))
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.policyToString(policy))
    # Execute policy (using epsilon greedy with epsilon = 0.0
    QLearning(mdp, EpsilonGreedy(epsilon = 0.0), qfunction = qfunction).execute(episodes = 2000)
    qLearningRewards = mdp.getRewards()

    print("=====\nSARSA: Cliffworld\n=====")    
    mdp = CliffWorld()
    qfunction = QTable()
    SARSA(mdp, EpsilonGreedy(epsilon = 0.2), qfunction).execute(episodes = 2000)
    print(mdp.qFunctionToString(qfunction))
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.policyToString(policy))
    #mdp.visualiseQFunction(qFunction, title="SARSA: Cliffworld", showText=True)
    # Execute policy (using epsilon greedy with epsilon = 0.0
    SARSA(mdp, EpsilonGreedy(epsilon = 0.0), qfunction = qfunction).execute(episodes = 2000)
    sarsaRewards = mdp.getRewards()

    from plot import Plot
    Plot.plotRewardsPerEpisode(["Q-learning", "SARSA"], [qLearningRewards, sarsaRewards])

    print("==========\nLinearSarsa: Gridworld one terminal state\n==========")
    from linear_qfunction import LinearQFunction
    mdp = GridWorld(discount_factor = 0.9, noise=0.1, goals=[((3,2),1)])
    features = GridWorldFeatureExtractor(mdp)
    qfunction = LinearQFunction(features)
    SARSA(mdp, EpsilonGreedy(), qfunction).execute(episodes = 2000)
    policy = mdp.extractPolicyFromQFunction(qfunction)
    print(mdp.qFunctionToString(qfunction))
    #mdp.visualiseQFunction(qFunction, title="LinearSarsa: Gridworld one terminal state", showText=True)
    print(mdp.policyToString(policy))
    '''
    print("==========\nLinear function approximation with Q-learning: Gridworld\n==========")
    mdp = GridWorld(width = width, height = height, goals = goals)
    from linear_qfunction import LinearQFunction
    features = GridWorldFeatureExtractor(mdp)
    qfunction = LinearQFunction(features)
    QLearning(mdp, EpsilonGreedy(), qfunction).execute(episodes = episodes)
    policy = mdp.extract_policy_from_q_function(qfunction)
    print(mdp.q_function_to_string(qfunction))
    print(mdp.policy_to_string(policy))
    linear_q_learning_rewards = mdp.get_rewards()
    
    print("=========\nQ-Learning with Reward shaping: Gridworld\n========")
    mdp = GridWorld(width = width, height = height, goals = goals)
    #mdp = GridWorld(discount_factor = 0.9, noise=0.1, goals=[((3,2),1)])
    #mdp = GridWorld(noise = 0.0, width = 6, height = 5, blockedStates = [], goals = [((5,4), 1), ((5,3),-1)])
    potential = GridWorldPotentialFunction(mdp)
    #qfunction = QTable()
    qfunction = LinearQFunction(features)
    RewardShapedQLearning(mdp, EpsilonGreedy(), potential, qfunction).execute(episodes = episodes)
    policy = mdp.extract_policy_from_q_function(qfunction)
    print(mdp.q_function_to_string(qfunction))
    print(mdp.policy_to_string(policy))
    reward_shaped_rewards = mdp.get_rewards()

    

    from plot import Plot
    Plot.plot_episode_length(["Tabular Q-learning", "Reward shaping", "Linear Q-Learning"], 
                            [q_learning_rewards, reward_shaped_rewards, linear_q_learning_rewards])
