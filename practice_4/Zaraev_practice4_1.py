import numpy as np
import gym
import matplotlib.pyplot as plt



def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)

def MonteCarlo(env, episode_n, trajectory_len=500, gamma=0.99):
    total_rewards = []

    state_n = env.observation_space.n
    action_n = env.action_space.n
    qfunction = np.zeros((state_n, action_n))
    counter = np.zeros((state_n, action_n))

    for episode in range(episode_n):
        epsilon = 1 - episode / episode_n
        trajectory = {'states': [], 'actions': [], 'rewards': []}

        state = env.reset()
        for _ in range(trajectory_len):
            trajectory['states'].append(state)

            action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
            trajectory['actions'].append(action)

            state, reward, done, _ = env.step(action)
            trajectory['rewards'].append(reward)

            if done:
                break

        total_rewards.append(sum(trajectory['rewards']))

        real_trajectory_len = len(trajectory['rewards'])
        returns = np.zeros(real_trajectory_len + 1)
        for t in range(real_trajectory_len - 1, -1, -1):
            returns[t] = trajectory['rewards'][t] + gamma * returns[t + 1]

        for t in range(real_trajectory_len):
            state = trajectory['states'][t]
            action = trajectory['actions'][t]
            qfunction[state][action] += (returns[t] - qfunction[state][action]) / (1 + counter[state][action])
            counter[state][action] += 1

    return total_rewards


def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)

    state_n = env.observation_space.n
    action_n = env.action_space.n
    qfunction = np.zeros((state_n, action_n))

    for episode in range(episode_n):
        epsilon = 1 / (episode + 1)

        state = env.reset()
        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for _ in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)
            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)

            qfunction[state][action] += alpha * (
                        reward + gamma * qfunction[next_state][next_action] - qfunction[state][action])

            state = next_state
            action = next_action

            total_rewards[episode] += reward

            if done:
                break

    return total_rewards

def QLearning(env, episode_n, noisy_episode_n, gamma=0.99, t_max=500, alpha=0.5):
    total_rewards = np.zeros(episode_n)

    state_n = env.observation_space.n
    action_n = env.action_space.n
    qfunction = np.zeros((state_n, action_n))

    for episode in range(episode_n):
        epsilon = 1 / (episode + 1)

        state = env.reset()
        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for _ in range(noisy_episode_n):
            next_state, reward, done, _ = env.step(action)
            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)
            qfunction[state][action] += alpha * (reward + gamma * max(qfunction[next_state]) - qfunction[state][action])

            state = next_state
            action = next_action

            total_rewards[episode] += reward
            if done:
                break

    return total_rewards

class CrossEntropyAgent():
    def __init__(self, state_n, action_n, policy_type = 'default'):
        self.laplase = 5
        self.state_n = state_n
        self.action_n = action_n
        self.policy_type = policy_type
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))
        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1

        for state in range(self.state_n):
            if np.sum(new_model[state]) > 0:
                new_model[state] /= np.sum(new_model[state])
            else:
                new_model[state] = self.model[state].copy()

        self.model = new_model
        return None

def get_trajectory(env, agent, max_len=1000, visualize=False):
    trajectory = {'states': [], 'actions': [], 'rewards': []}

    obs = env.reset(return_info=True)[0]
    state = obs

    for _ in range(max_len):
        trajectory['states'].append(state)

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        obs, reward, done, _ = env.step(action)
        trajectory['rewards'].append(reward)

        state = obs

        if visualize:
            time.sleep(0.001)
            env.render()

        if done:
            break

    return trajectory

ACTION_N = 6
STATE_N = 500
env = gym.make("Taxi-v3")


total_rewards_monte = MonteCarlo(env, episode_n=500, trajectory_len=1000, gamma=0.99)
total_rewards_sarsa = SARSA(env, episode_n=500, trajectory_len=1000, gamma=0.999, alpha=0.5)
total_rewards_q = QLearning(env, episode_n=500, noisy_episode_n=400, t_max=1000, gamma=0.999, alpha=0.5)

agent = CrossEntropyAgent(state_n=STATE_N,
                          action_n=ACTION_N)
q_param = 0.6
iteration_n = 500
trajectory_n = 500

mean_total_rewards = []

for iteration in range(iteration_n):

    #policy evaluation
    trajectories = [get_trajectory(env, agent) for _ in range(trajectory_n)]
    total_rewards = [np.sum(trajectory['rewards']) for trajectory in trajectories]
    print('iteration:', iteration, 'mean total reward:', np.mean(total_rewards))
    mean_total_rewards.append(np.mean(total_rewards))

    #policy improvement
    quantile = np.quantile(total_rewards, q_param)
    elite_trajectories = []
    for trajectory in trajectories:
        total_reward = np.sum(trajectory['rewards'])
        if total_reward > quantile:
            elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)

plt.plot(total_rewards_monte)
plt.plot(total_rewards_q)
plt.plot(total_rewards_sarsa)
plt.plot(mean_total_rewards)
plt.xlabel('Iterations')
plt.legend(['Monte', 'SARSA','Q-learning', 'CrossEntropy'])
plt.ylabel('Reward')
plt.savefig('Check_init.jpg')
plt.show()