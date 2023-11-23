import gym
from tqdm import tqdm
import torch
from torch import nn
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

ACTION_N = 2
def default_dict(init_value):
    return [init_value]* ACTION_N

def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon
    return np.random.choice(np.arange(action_n), p=policy)


def discretize_state(state):
    cart_position_bins = np.linspace(-4.8, 4.8, 100)
    cart_velocity_bins = np.linspace(-100, 100, 1000)
    pole_angle_bins = np.linspace(-0.42, 0.42, 100)
    pole_velocity_bins = np.linspace(-100, 100, 1000)

    cart_pos, cart_vel, pole_ang, pole_vel = state

    discrete_state = (np.digitize(cart_pos, cart_position_bins),
                      np.digitize(cart_vel, cart_velocity_bins),
                      np.digitize(pole_ang, pole_angle_bins),
                      np.digitize(pole_vel, pole_velocity_bins))
    return discrete_state


def MonteCarlo(env, episode_n, trajectory_len=500, gamma=0.99) -> tuple[list[float], list[int]]:
    total_rewards = []
    trajectories_len = []

    action_n = env.action_space.n
    qfunction = defaultdict(lambda: [0 for _ in range(action_n)])
    counter = defaultdict(lambda: [0 for _ in range(action_n)])

    for episode in tqdm(range(episode_n)):
        epsilon = 1 - episode / episode_n
        trajectory = {'states': [], 'actions': [], 'rewards': []}

        state = env.reset()
        for trjct_idx in range(trajectory_len):
            state_round = discretize_state(state)  # , qfunction, action_n)
            trajectory['states'].append(state_round)

            action = get_epsilon_greedy_action(qfunction[state_round], epsilon, action_n)
            trajectory['actions'].append(action)

            state, reward, done, _ = env.step(action)
            trajectory['rewards'].append(reward)

            if done:
                break

        total_rewards.append(sum(trajectory['rewards']))
        trajectories_len.append(trjct_idx)

        real_trajectory_len = len(trajectory['rewards'])
        returns = np.zeros(real_trajectory_len + 1)
        for t in range(real_trajectory_len - 1, -1, -1):
            returns[t] = trajectory['rewards'][t] + gamma * returns[t + 1]

        for t in range(real_trajectory_len):
            state = trajectory['states'][t]
            action = trajectory['actions'][t]
            qfunction[state][action] += (returns[t] - qfunction[state][action]) / (1 + counter[state][action])
            counter[state][action] += 1

    return total_rewards, trajectories_len


def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    action_n = env.action_space.n

    qfunction = defaultdict(lambda: default_dict(init_value=0))

    total_rewards = []
    for episode in range(episode_n):
        total_reward = 0
        epsilon = 1 - episode / episode_n
        state = env.reset()

        state = discretize_state(state)

        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for t in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)

            next_state = discretize_state(next_state)

            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)
            total_reward += reward
            qfunction[state][action] += alpha * (
                        reward + gamma * qfunction[next_state][next_action] - qfunction[state][action])
            state = next_state
            action = next_action
            if done:
                break

        total_rewards.append(total_reward)

    return total_rewards


def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    action_n = env.action_space.n

    qfunction = defaultdict(lambda: default_dict(init_value=0))

    total_rewards = []
    for episode in range(episode_n):
        total_reward = 0
        epsilon = 1 - episode / episode_n
        state = env.reset()

        state = discretize_state(state)

        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for t in range(trajectory_len):
            next_state, reward, done, _ = env.step(action)

            next_state = discretize_state(next_state)

            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)
            total_reward += reward
            qfunction[state][action] += alpha * (
                        reward + gamma * qfunction[next_state][next_action] - qfunction[state][action])
            state = next_state
            action = next_action
            if done:
                break

        total_rewards.append(total_reward)

    return total_rewards


def QLearning(env, episode_n, t_max=1000, gamma=0.99, alpha=0.5):
    action_n = env.action_space.n

    qfunction = defaultdict(lambda: default_dict(init_value=0))

    total_rewards = []
    for episode in range(episode_n):
        total_reward = 0
        epsilon = 1 - episode / episode_n
        state = env.reset()

        state = discretize_state(state)

        action = get_epsilon_greedy_action(qfunction[state], epsilon, action_n)
        for t in range(t_max):
            next_state, reward, done, _ = env.step(action)

            next_state = discretize_state(next_state)

            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon, action_n)

            total_reward += reward

            qfunction[state][action] += alpha * (reward + gamma * max(qfunction[next_state]) - qfunction[state][action])

            state = next_state
            action = next_action
            if done:
                break

        total_rewards.append(total_reward)

    return total_rewards

class CEM(nn.Module):
    def __init__(self, state_dim, action_n, lr):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.lr = lr

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, self.action_n)
        )

        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        action_prob = self.softmax(logits).detach().numpy()
        action = np.random.choice(self.action_n, p=action_prob)
        return action

    def update_policy(self, elite_trajectories):
        elite_states = []
        elite_actions = []
        for trajectory in elite_trajectories:
            elite_states.extend(trajectory['states'])
            elite_actions.extend(trajectory['actions'])
        elite_states = torch.FloatTensor(elite_states)
        elite_actions = torch.LongTensor(elite_actions)

        loss = self.loss(self.forward(elite_states), elite_actions)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

def get_trajectory(env, agent, trajectory_len, visualize=False):
    trajectory = {'states': [], 'actions': [], 'total_reward': 0}

    state = env.reset()
    trajectory['states'].append(state)

    for _ in range(trajectory_len):

        action = agent.get_action(state)
        trajectory['actions'].append(action)

        state, reward, done, _ = env.step(action)
        trajectory['total_reward'] += reward

        if done:
            break

        if visualize:
            env.render()

        trajectory['states'].append(state)
    return trajectory


def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param)
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]

env = gym.make("CartPole-v1")
total_rewards_monte = MonteCarlo(env, episode_n=10_000, trajectory_len=1000, gamma=0.99)
total_rewards_sarsa = SARSA(env, episode_n=10_000, trajectory_len=10000, gamma=0.999, alpha=0.5)
total_rewards_q = QLearning(env, episode_n=10_000,  t_max=1000, gamma=0.999, alpha=0.5)

state_dim = 4
lr = 1e-2

agent = CEM(state_dim, ACTION_N, lr)
episode_n = 1_000
trajectory_n = 50
trajectory_len = 500
q_param = 0.9
mean_total_rewards = []

for episode in tqdm(range(episode_n)):
    trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]

    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    mean_total_rewards.append(mean_total_reward)
#     print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')

    elite_trajectories = get_elite_trajectories(trajectories, q_param)

    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)

plt.plot(total_rewards_monte)
plt.plot(total_rewards_q)
plt.plot(total_rewards_sarsa)
plt.xlabel('Iterations')
plt.legend(['Monte', 'SARSA','Q-learning'])
plt.ylabel('Reward')
plt.savefig('Check_practice2.jpg')
plt.show()