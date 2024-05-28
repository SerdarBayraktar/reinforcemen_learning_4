import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from reinforcemen_learning_4.agent import BoltzmanPolicy, ThompsonPolicy, SARSAAgent,EgreedyPolicy
from reinforcemen_learning_4.dynamic_env import DynamicMazeEnvironment
from reinforcemen_learning_4.environment import ShortcutEnvironment,BanditEnvironment
from Helper import smooth,LearningCurvePlot

def run_repetitions_greedy(n_actions, n_timesteps, n_repetitions, smoothing_window, epsilon=0.1):
    rewards = np.zeros((n_repetitions, n_timesteps))

    for i in range(n_repetitions):  # using multple to get a avarage value.
        env = BanditEnvironment(n_actions)
        policy = EgreedyPolicy(n_actions)
        for t in range(n_timesteps):  # single bandit experiment
            print(t)
            action = policy.select_action(epsilon)
            reward = env.act(action)
            policy.update(action, reward)
            rewards[i, t] = reward
    avg_rewards = np.mean(rewards, axis=0)
    smooth_rewards = smooth(avg_rewards, smoothing_window, poly=2)
    return smooth_rewards

def run_repetitions_thompson(n_actions, n_timesteps, n_repetitions, smoothing_window, epsilon=0.1):
    rewards = np.zeros((n_repetitions, n_timesteps))

    for i in range(n_repetitions):  # using multple to get a avarage value.
        env = BanditEnvironment(n_actions)
        policy = ThompsonPolicy(n_actions)
        for t in range(n_timesteps):  # single bandit experiment
            print(t)
            action = policy.select_action(epsilon)
            reward = env.act(action)
            policy.update(action, reward)
            rewards[i, t] = reward
    avg_rewards = np.mean(rewards, axis=0)
    smooth_rewards = smooth(avg_rewards, smoothing_window, poly=2)
    return smooth_rewards


def run_repetitions_sarsa(n_rep, n_episodes, alpha=0.1, epsilon=0.1, gamma=1,):
    rewards = np.zeros((n_rep, n_episodes))

    for rep in range(n_rep):
        agent = SARSAAgent(n_actions=4, n_states=144,
                           epsilon=epsilon, alpha=alpha, gamma=gamma)
        environment = ShortcutEnvironment()
        for episode in range(n_episodes):
            print(episode)
            state = environment.reset()
            terminal = False
            total_reward = 0
            action = agent.select_action(state)

            while not terminal:
                reward = environment.step(action,episode)
                next_state = environment.state()
                terminal = environment.done()
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                total_reward += reward

            rewards[rep, episode] = total_reward
        #if n_rep == 1:
        #    print_greedy_actions(agent.Q)
        if rep % 20 == 0:
            print(rep)
    avg_cumulative_rewards = np.mean(rewards, axis=0)
    # return savgol_filter(avg_cumulative_rewards, 31, 1)
    return avg_cumulative_rewards

def run_repetitions_sarsa_dynamic(n_rep, n_episodes, alpha=0.1, epsilon=0.1, gamma=1,):
    rewards = np.zeros((n_rep, n_episodes))

    for rep in range(n_rep):
        agent = SARSAAgent(n_actions=4, n_states=144,
                           epsilon=epsilon, alpha=alpha, gamma=gamma)
        environment = DynamicMazeEnvironment()
        for episode in range(n_episodes):
            print(episode)
            state = environment.reset()
            terminal = False
            total_reward = 0
            action = agent.select_action(state)

            while not terminal:
                reward = environment.step(action)
                next_state = environment.state()
                terminal = environment.done()
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                total_reward += reward

            rewards[rep, episode] = total_reward
        #if n_rep == 1:
        #    print_greedy_actions(agent.Q)
        if rep % 20 == 0:
            print(rep)
    avg_cumulative_rewards = np.mean(rewards, axis=0)
    # return savgol_filter(avg_cumulative_rewards, 31, 1)
    return avg_cumulative_rewards

def run_repetitions_boltzman(n_rep, n_episodes, alpha=0.1, ):
    cumulative_rewards = np.zeros((n_rep, n_episodes))

    for rep in range(n_rep):
        # Initialize agent and environment
        agent = BoltzmanPolicy(n_actions=4,n_states=144)
        # Assuming this is how you initialize your environment

        environment = ShortcutEnvironment()

        for episode in range(n_episodes):
            print(episode)
            # Reset environment and get initial state
            state = environment.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(state)
                reward = environment.step(action,episode)
                next_state = environment.state()
                done = environment.done()
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                total_reward += reward

            cumulative_rewards[rep, episode] = total_reward

    if n_rep > 1:
        avg_cumulative_rewards = np.mean(cumulative_rewards, axis=0)
        return avg_cumulative_rewards
    else:
        # Return the cumulative rewards for the single repetition
        return cumulative_rewards[0]

def run_repetitions_boltzman_dynamic(n_rep, n_episodes, alpha=0.1, ):
    cumulative_rewards = np.zeros((n_rep, n_episodes))

    for rep in range(n_rep):
        # Initialize agent and environment
        agent = BoltzmanPolicy(n_actions=4,n_states=144)
        # Assuming this is how you initialize your environment

        environment = DynamicMazeEnvironment()

        for episode in range(n_episodes):
            print(episode)
            # Reset environment and get initial state
            state = environment.reset()
            done = False
            total_reward = 0

            while not done:
                action = agent.select_action(state)
                reward = environment.step(action)
                next_state = environment.state()
                done = environment.done()
                next_action = agent.select_action(next_state)
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                total_reward += reward

            cumulative_rewards[rep, episode] = total_reward

    if n_rep > 1:
        avg_cumulative_rewards = np.mean(cumulative_rewards, axis=0)
        return avg_cumulative_rewards
    else:
        # Return the cumulative rewards for the single repetition
        return cumulative_rewards[0]

def plot_learning_curve(curves, labels, xlabel, ylabel, title, save_path):
    if len(curves) != len(labels):
        raise ValueError("Number of curves and labels must match",
                         len(curves), len(labels), labels, xlabel, ylabel, title, save_path)

    for i, avg_cumulative_rewards in enumerate(curves):
        # smooth it.
        smoothed_rewards = savgol_filter(
            avg_cumulative_rewards, window_length=31, polyorder=1)
        plt.plot(smoothed_rewards, label=labels[i])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def experiment_boltzman():
    print('Sarsa normal environment single run 10000 episode')
    single_sarsa = run_repetitions_sarsa(10, 1000)
    avg_rewards_single = run_repetitions_boltzman(10, 1000)
    plot_learning_curve([single_sarsa,avg_rewards_single,],
                        ['E greedy exploration','Boltzman exploration',], 'Episodes',
                        'Average Cumulative Reward', 'Learning Curve', 'sarsa_boltzman.png')
    #plot_learning_curve([avg_rewards_single], ['Boltzman'], 'Episodes',
    #                    'Average Cumulative Reward', 'Learning Curve', 'boltzman.png')

def experiment_thompson():
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    epsilons = [0.1]# best selected
    reward_plot = LearningCurvePlot(title="Egreedy Policy Curve")
    for epsilon in epsilons:
        smooth_rewards_greedy = run_repetitions_greedy(n_actions, n_timesteps, n_repetitions, smoothing_window, epsilon)
        smooth_rewards_thompson = run_repetitions_thompson(n_actions, n_timesteps, n_repetitions, smoothing_window, epsilon)
        plot_learning_curve([smooth_rewards_greedy, smooth_rewards_thompson], ['E-greedy 0.1', 'Thompson'], 'Time',
                            'Reward', 'Learning Curve', 'greedy_thompspn.png')

    reward_plot.save(name='thompson.png')
def experiment():

    experiment_boltzman()
    experiment_thompson()


if __name__ == '__main__':
    experiment()