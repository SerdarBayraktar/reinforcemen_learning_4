#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment

class EgreedyPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.Q = np.zeros(n_actions)
        self.n = np.zeros(n_actions)

        pass
        
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
        
    def update(self,a,r):
        self.n[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.n[a]
        pass

class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.means = np.full(n_actions, initial_value)
        pass
        
    def select_action(self):
        # Exploitation: choose the action with the highest estimated mean
        a = np.argmax(self.means)
        return a
        
    def update(self,a,r):
        # Update the mean value of action 'a'
        self.means[a] += self.learning_rate * (r - self.means[a])
        pass

class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.means = np.zeros(n_actions)
        self.counts = np.zeros(n_actions)
        pass
    
    def select_action(self, c, t):
        # untried actions will always be selected
        untried_actions = self.counts == 0
        if np.any(untried_actions):
            # if there are any untried actions, select one randomly
            return np.random.choice(np.where(untried_actions)[0])
        
        a = np.argmax(self.means + c * np.sqrt(np.log(t) / (self.counts)))
        return a
        
    def update(self,a,r):
        # TO DO: Add own code
        self.counts[a] += 1
        self.means[a] += (r - self.means[a]) / self.counts[a]
        pass
    
def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
    
    pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(epsilon=0.9) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print(pi.Q)
    print("Test e-greedy policy with action {}, received reward {}".format(a,r))

    pi = OIPolicy(n_actions=n_actions,initial_value=1.0) # Initialize policy
    a = pi.select_action() # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a,r))
    
    pi = UCBPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(c=1.0,t=1) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test UCB policy with action {}, received reward {}".format(a,r))

if __name__ == '__main__':
    test()
