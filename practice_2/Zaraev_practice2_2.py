import torch
from torch import nn
import numpy as np
import gym
import random
import matplotlib.pyplot as plt


class CEM(nn.Module):
    def __init__(self, state_dim, action_n, lr):
        super().__init__()
        self.state_dim = state_dim
        self.action_n = action_n
        self.lr = lr
        self.tanh = nn.Tanh()

        self.network = nn.Sequential(
            nn.Linear(self.state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

        self.softmax = nn.Softmax()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, _input):
        return self.network(_input)

    def get_action(self, state):
        state = torch.FloatTensor(state)
        logits = self.forward(state)
        action_prob = self.tanh(logits).detach().numpy()
        # action = np.random.choice(self.action_n, p=action_prob)
        action = logits.item()
        return action

    def get_action_uniform(self):
        return np.random.uniform(-1,1)

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
    epsilon = 1
    epsilon_step = .0025

    state = env.reset()
    trajectory['states'].append(state)

    for _ in range(trajectory_len):
        if random.random() > epsilon:
            action = agent.get_action(state)
            trajectory['actions'].append(action)
        else:
            action = agent.get_action_uniform()
            trajectory['actions'].append(action)

        state, reward, done, _ = env.step(np.array([action]))
        trajectory['total_reward'] += reward

        if done:
            break

        if visualize:
            env.render()

        epsilon -= epsilon_step

        trajectory['states'].append(state)
    return trajectory


def get_elite_trajectories(trajectories, q_param):
    total_rewards = [trajectory['total_reward'] for trajectory in trajectories]
    quantile = np.quantile(total_rewards, q=q_param)
    return [trajectory for trajectory in trajectories if trajectory['total_reward'] > quantile]


env = gym.make('MountainCarContinuous-v0')
state_dim = 2
action_n = 3
lr = 1e-1

agent = CEM(state_dim, action_n, lr)
episode_n = 50
trajectory_n = 50
trajectory_len = 1000
q_param = 0.6
mean_total_rewards = []

for episode in range(episode_n):
    trajectories = [get_trajectory(env, agent, trajectory_len) for _ in range(trajectory_n)]

    mean_total_reward = np.mean([trajectory['total_reward'] for trajectory in trajectories])
    mean_total_rewards.append(mean_total_reward)
    print(f'episode: {episode}, mean_total_reward = {mean_total_reward}')

    elite_trajectories = get_elite_trajectories(trajectories, q_param)

    if len(elite_trajectories) > 0:
        agent.update_policy(elite_trajectories)

get_trajectory(env, agent, trajectory_len, visualize=True)

plt.plot(range(episode_n), mean_total_rewards)
plt.legend(['Cross Entropy'])
plt.xlabel('Iterations')
plt.title(f'q_param - {q_param} iteration_n : {episode_n} trajectory_n : {trajectory_n} max_reward : {max(mean_total_rewards)}')
plt.ylabel('Reward')
plt.savefig('Policy_best.jpg')
plt.show()