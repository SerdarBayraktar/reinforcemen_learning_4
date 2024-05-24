import random
import numpy as np


class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1, gamma=1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # Initialize action values Q(s, a) to 0
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        # Implement epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        self.Q[state][action] += self.alpha * \
            (td_target - self.Q[state][action])


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
        # TO DO: Add own code
        # Replace this with correct action selection
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.n_actions))
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, next_action):
        # TO DO: Add own code
        self.Q[state][action] += self.alpha * \
            (reward + self.gamma * self.Q[next_state]
             [next_action] - self.Q[state][action])


class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha, gamma):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state):
        if random.random() > self.epsilon:  # Exploitation
            return np.argmax(self.Q[state])
        else:  # Exploration
            return random.choice(range(self.n_actions))

    def update(self, state, action, reward, next_state):
        # Compute expected Q value for the next state
        expected_Q_next = np.dot(
            self.Q[next_state], self._get_action_probabilities(next_state))

        # SARSA Update
        self.Q[state, action] += self.alpha * \
            (reward + self.gamma * expected_Q_next - self.Q[state, action])

    def _get_action_probabilities(self, state):
        probabilities = np.ones(self.n_actions) * self.epsilon / self.n_actions
        best_action = np.argmax(self.Q[state])
        probabilities[best_action] += (1.0 - self.epsilon)
        return probabilities
