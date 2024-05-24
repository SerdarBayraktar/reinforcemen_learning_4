import numpy as np
class BoltzmanPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.n = np.zeros(n_actions)

        pass

    def select_action(self, epsilon):
        #boltzman
        pass

    def update(self, a, r):
        self.n[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.n[a]
        pass


class ThompsonPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.means = np.full(n_actions, initial_value)
        pass

    def select_action(self):
        # thompspson
        pass

    def update(self, a, r):
        # Update the mean value of action 'a'
        #self.means[a] += self.learning_rate * (r - self.means[a])
        pass