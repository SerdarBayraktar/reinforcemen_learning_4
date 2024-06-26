import random
import numpy as np


class BanditEnvironment:

    def __init__(self, n_actions):
        ''' Initializes a bandit environment'''
        self.n_actions = n_actions
        self.means = np.random.uniform(low=0.0, high=1.0, size=n_actions)
        # self.means = np.random.normal(loc=0.0,scale=1.0,size=n_actions) # alternative bandit definitions are possible
        self.best_action = np.argmax(self.means)
        self.best_average_return = np.max(self.means)

    def act(self, a):
        ''' returns a sampled reward for action a '''
        r = np.random.binomial(1, self.means[a])
        # r= np.random.normal(loc=self.means[a],scale=1.0) # alternative bandit definitions are possible
        return r


class Environment(object):

    def __init__(self):
        pass

    def reset(self,episode):
        '''Reset the environment.

        Returns:
           starting_position: Starting position of the agent.
        '''
        raise Exception("Must be implemented by subclass.")

    def render(self):
        '''Render environment to screen.'''
        raise Exception("Must be implemented by subclass.")

    def step(self, action):
        '''Take action.

        Arguments:
           action: action to take.

        Returns:
           reward: reward of action taken.
        '''
        raise Exception("Must be implemented by subclass.")

    def possible_actions(self):
        '''Return list of possible actions in current state.

        Returns:
          actions: list of possible actions.
        '''
        raise Exception("Must be implemented by subclass.")

    def state(self):
        '''Return current state.

        Returns:
          state: environment-specific representation of current state.
        '''
        raise Exception("Must be implemented by subclass.")

    def state_size(self):
        '''Return the number of elements of the state space.

        Returns:
          state_size: number of elements of the state space.
        '''
        raise Exception("Must be implemented by subclass.")

    def action_size(self):
        '''Return the number of elements of the action space.

        Returns:
          state_size: number of elements of the action space.
        '''
        raise Exception("Must be implemented by subclass.")

    def done(self):
        '''Return whether current episode is finished and environment should be reset.

        Returns:
          done: True if current episode is finished.
        '''
        raise Exception("Must be implemented by subclass.")


class ShortcutEnvironment(Environment):
    def __init__(self, seed=None):
        self.r = 12
        self.c = 12
        self.rng = random.Random(seed)
        s = np.zeros((self.r, self.c + 1), dtype=str)
        s[:] = 'X'
        s[:, -1] = '\n'
        s[self.r // 3:, self.c // 3:2 * self.c // 3] = 'C'
        s[5 * self.r // 6 - 1, :self.c // 2] = 'X'
        s[2 * self.r // 3:5 * self.r // 6:, self.c // 2] = 'X'
        s[2 * self.r // 3, self.c // 2:2 * self.c // 3] = 'X'
        s[2 * self.r // 3, 2 * self.c // 3] = 'G'
        self.s = s
        self.reset()

    def reset(self):
        self.x = self.c // 6
        rand_number = int(2 * self.rng.random())
        if rand_number:
            self.y = 5 * self.r // 6 - 1
        else:
            self.y = self.r // 6
        self.starty = self.y
        self.isdone = False
        return rand_number

    def state(self):
        return self.y * self.c + self.x

    def state_size(self):
        return self.c * self.r

    def action_size(self):
        return 4

    def done(self):
        return self.isdone

    def possible_actions(self):
        return [0, 1, 2, 3]

    def step(self, action,episode):
        if self.isdone:
            raise ValueError('Environment has to be reset.')

        if not action in self.possible_actions():
            raise ValueError(f'Action ({action}) not in set of possible actions.')

        if episode == 500:
            s = self.s
            s[9,4] = 'C'
            self.s = s
        if episode == 600:
            s = self.s
            s[9,4] = 'C'
            s[9, 5] = 'X'
            self.s = s
        if episode == 700:
            s = self.s
            s[9,5] = 'C'
            s[9, 6] = 'X'
            self.s = s

        if action == 0:
            if self.y > 0:
                self.y -= 1
        elif action == 1:
            if self.y < self.r - 1:
                self.y += 1
        elif action == 2:
            if self.x > 0:
                self.x -= 1
        elif action == 3:
            if self.x < self.c - 1:
                self.x += 1

        if self.s[self.y, self.x] == 'G':  # Goal reached
            self.isdone = True
            return -1
        elif self.s[self.y, self.x] == 'C':  # Fall off cliff
            self.x = self.c // 6
            self.y = self.starty
            return -100
        return -1

    def render(self):
        s = self.s.copy()
        s[self.y, self.x] = 'p'
        print(s.tobytes().decode('utf-8'))

