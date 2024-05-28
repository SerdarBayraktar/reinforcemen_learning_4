import numpy as np
import random


class DynamicMazeEnvironment:
    def __init__(self, size=12, obstacle_rate=0.2, change_frequency=0.1):
        self.size = size
        self.obstacle_rate = obstacle_rate
        self.change_frequency = change_frequency
        self.grid = np.zeros((size, size))
        self.position = (random.randint(0, size - 1), random.randint(0, size - 1))
        self.goal = self.generate_new_position(exclude=self.position)
        self.initialize_grid()

    def initialize_grid(self):
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < self.obstacle_rate:
                    self.grid[i, j] = 1  # 1 represents an obstacle

    def generate_new_position(self, exclude=None):
        while True:
            position = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if position != exclude and self.grid[position[0], position[1]] == 0:
                return position

    def step(self, action):
        old_x, old_y = self.position
        if action == 0 and old_y > 0:  # Move up
            new_y = old_y - 1
        elif action == 1 and old_y < self.size - 1:  # Move down
            new_y = old_y + 1
        else:
            new_y = old_y

        if action == 2 and old_x > 0:  # Move left
            new_x = old_x - 1
        elif action == 3 and old_x < self.size - 1:  # Move right
            new_x = old_x + 1
        else:
            new_x = old_x

        if self.grid[new_x, new_y] == 1:
            new_x, new_y = old_x, old_y  # Hit an obstacle, don't move

        self.position = (new_x, new_y)
        reward = -1  # Default reward is -1 per move
        done = False

        if (new_x, new_y) == self.goal:
            reward = 100  # Reward for reaching the goal
            done = True
        elif (new_x, new_y) == (old_x, old_y):
            reward = -5  # Penalty for hitting an obstacle

        self.update_environment()

        return reward

    def update_environment(self):
        if random.random() < self.change_frequency:
            # Update obstacles and possibly the goal position
            x, y = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            # Toggle the obstacle state safely by directly setting it
            if self.grid[x, y] == 1:
                self.grid[x, y] = 0  # Remove obstacle if present
            else:
                self.grid[x, y] = 1  # Add obstacle if not present
            self.goal = self.generate_new_position(exclude=self.position)

    def render(self):
        display_grid = np.where(self.grid == 1, '█', ' ')
        display_grid[self.position[0], self.position[1]] = 'P'
        display_grid[self.goal[0], self.goal[1]] = 'G'
        print("\n".join(["".join(row) for row in display_grid]))

    def reset(self):
        self.position = self.generate_new_position()
        self.goal = self.generate_new_position(exclude=self.position)
        self.initialize_grid()
        return self.state()

    def state(self):
        x, y = self.position
        return y * self.size + x

    def possible_actions(self):
        # Return all possible actions (0: up, 1: down, 2: left, 3: right)
        return [0, 1, 2, 3]
    def done(self):
        return self.position == self.goal