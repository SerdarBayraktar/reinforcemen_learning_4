import numpy as np
import random

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1, gamma=1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # TO DO: Add own code

        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))
        pass

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action):
        self.Q[state][action] += self.alpha * \
            (reward + self.gamma * self.Q[next_state]
             [next_action] - self.Q[state][action])

class BoltzmanPolicy:

    def __init__(self, n_actions=4, n_states=144,t=1.0, alpha=0.1):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.n = np.zeros(n_actions)
        self.t = t
        self.alpha = alpha
        self.Q = np.zeros((n_states, n_actions))
        pass

    def select_action(self, state):
        exp_Q = np.exp(self.Q[state] /self.t)
        probabilities = exp_Q / np.sum(exp_Q)
        action = np.random.choice(self.n_actions, p=probabilities)
        return action

    def update(self, state, action, reward, next_state, next_action):
        self.Q[state][action] += self.alpha * \
                                 (reward + 1 * self.Q[next_state] # gamma is the constant
                                 [next_action] - self.Q[state][action])

class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.n = np.zeros(n_actions)

    def select_action(self, epsilon):
        if np.random.rand() < epsilon:
            others = []
            for a in range(self.n_actions):
                if a != np.argmax(self.Q):
                    others.append(a)
            action = np.random.choice(others)
        else:
            action = np.argmax(self.Q)
        return action

    def update(self, a, r):
        self.n[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.n[a]
        pass


class ThompsonPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.n = np.zeros(n_actions)
        self.successes = np.zeros(n_actions)
        self.failures = np.zeros(n_actions)

    def select_action(self, epsilon):
        sampled_theta = [np.random.beta(self.successes[a] + 1, self.failures[a] + 1) for a in range(self.n_actions)]
        return np.argmax(sampled_theta)

    def update(self, a, r):
        self.n[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.n[a]
        if r > 0:
            self.successes[a] = self.successes[a] + 1
        else:
            self.failures[a] += 1
