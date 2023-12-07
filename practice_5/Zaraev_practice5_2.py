import numpy as np
import random
import torch
import torch.nn as nn
import gym
import copy
from collections import OrderedDict
from typing import List
import matplotlib.pyplot as plt


class Qfunction(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.linear_1 = nn.Linear(state_dim, 64)
        self.linear_2 = nn.Linear(64, 64)
        self.linear_3 = nn.Linear(64, action_dim)
        self.activation = nn.ReLU()

    def forward(self, states):
        hidden = self.linear_1(states)
        hidden = self.activation(hidden)
        hidden = self.linear_2(hidden)
        hidden = self.activation(hidden)
        actions = self.linear_3(hidden)
        return actions


class DQN:
    def __init__(self, state_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=64, epsilon_decrease=0.01,
                 epilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = []
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def _fit_one_step(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()

            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease

    def fit(self, env, episode_n, t_max=500) -> List[float]:
        totals_rewards = []
        for _ in range(episode_n):
            total_reward = 0

            state = env.reset()
            for _ in range(t_max):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self._fit_one_step(state, action, reward, done, next_state)

                state = next_state
                total_reward += reward

                if done:
                    break

            totals_rewards.append(total_reward)

        return totals_rewards

class DQNHardTargetUpdate:
    def __init__(
            self,
            state_dim,
            action_dim,
            gamma=0.99,
            lr=1e-3,
            batch_size=64,
            epsilon_decrease=0.01,
            epilon_min=0.01,
            q_fix_update_epoch: int = None,
            q_fix_update_step: int = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix.load_state_dict(copy.deepcopy(self.q_function.state_dict()))
        self.q_fix_update_epoch = q_fix_update_epoch
        self.q_fix_update_step = q_fix_update_step
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = []
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def _fit_one_step(self, state, action, reward, done, next_state, step):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function_fix(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()

            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease

            if self.q_fix_update_step is not None and (step + 1) % self.q_fix_update_step == 0:
                self.q_function_fix = self.q_function

    def fit(self, env, episode_n, t_max: int = 500):
        totals_rewards = []
        for episode_idx in range(episode_n):
            total_reward = 0

            state = env.reset()
            for step in range(t_max):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self._fit_one_step(state, action, reward, done, next_state, int((episode_idx + 1) * step))

                state = next_state
                total_reward += reward

                if done:
                    break

            if self.q_fix_update_epoch is not None and (episode_idx + 1) % self.q_fix_update_epoch == 0:
                self.q_function_fix = self.q_function

            totals_rewards.append(total_reward)

        return totals_rewards


class DQNSoftTargetUpdate:
    def __init__(
            self,
            state_dim,
            action_dim,
            gamma=0.99,
            lr=1e-3,
            batch_size=64,
            epsilon_decrease=0.01,
            epilon_min=0.01,
            tau: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix.load_state_dict(copy.deepcopy(self.q_function.state_dict()))
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = []
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def _mix_weights(self, ):
        """
        Updated model weight for q_function_fix.
        """
        new_weights = []
        for (name_1, param_1), (name_2, param_2) in zip(self.q_function.named_parameters(),
                                                        self.q_function_fix.named_parameters()):
            assert name_1 == name_2, "Model structure is not identical"
            new_weights.append(
                [
                    name_2,
                    self.tau * param_1.data + (1 - self.tau) * param_2.data
                ]
            )
        self.q_function_fix.load_state_dict(OrderedDict(new_weights))

    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def _fit_one_step(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            targets = rewards + self.gamma * (1 - dones) * torch.max(self.q_function_fix(next_states), dim=1).values
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()

            self._mix_weights()

            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease

    def fit(self, env, episode_n, t_max: int = 500) -> List[float]:
        totals_rewards = []
        for _ in range(episode_n):
            total_reward = 0

            state = env.reset()
            for _ in range(t_max):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self._fit_one_step(state, action, reward, done, next_state)

                state = next_state
                total_reward += reward

                if done:
                    break

            totals_rewards.append(total_reward)

        return totals_rewards


class DoubleDQN:

    def __init__(
            self,
            state_dim,
            action_dim,
            gamma=0.99,
            lr=1e-3,
            batch_size=64,
            epsilon_decrease=0.01,
            epilon_min=0.01,
            tau: float = 0.01,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_function = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix = Qfunction(self.state_dim, self.action_dim)
        self.q_function_fix.load_state_dict(copy.deepcopy(self.q_function.state_dict()))
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = 1
        self.epsilon_decrease = epsilon_decrease
        self.epilon_min = epilon_min
        self.memory = []
        self.optimzaer = torch.optim.Adam(self.q_function.parameters(), lr=lr)

    def _mix_weights(self, ):
        """
        Updated model weight for q_function_fix.
        """
        new_weights = []
        for (name_1, param_1), (name_2, param_2) in zip(self.q_function.named_parameters(),
                                                        self.q_function_fix.named_parameters()):
            assert name_1 == name_2, "Model structure is not identical"
            new_weights.append(
                [
                    name_2,
                    self.tau * param_1.data + (1 - self.tau) * param_2.data
                ]
            )
        self.q_function_fix.load_state_dict(OrderedDict(new_weights))

    def get_action(self, state):
        q_values = self.q_function(torch.FloatTensor(state))
        argmax_action = torch.argmax(q_values)
        probs = self.epsilon * np.ones(self.action_dim) / self.action_dim
        probs[argmax_action] += 1 - self.epsilon
        action = np.random.choice(np.arange(self.action_dim), p=probs)
        return action

    def _fit_one_step(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, int(done), next_state])

        if len(self.memory) > self.batch_size:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, dones, next_states = map(torch.tensor, list(zip(*batch)))

            argmax_actions_from_q_fix = torch.argmax(self.q_function_fix(next_states), dim=1)
            targets = (
                    rewards
                    + self.gamma
                    * (1 - dones)
                    * self.q_function(next_states)[torch.arange(self.batch_size), argmax_actions_from_q_fix]
            )
            q_values = self.q_function(states)[torch.arange(self.batch_size), actions]

            loss = torch.mean((q_values - targets.detach()) ** 2)
            loss.backward()
            self.optimzaer.step()
            self.optimzaer.zero_grad()

            self._mix_weights()

            if self.epsilon > self.epilon_min:
                self.epsilon -= self.epsilon_decrease

    def fit(self, env, episode_n, t_max: int = 500) -> List[float]:
        totals_rewards = []
        for _ in range(episode_n):
            total_reward = 0

            state = env.reset()
            for _ in range(t_max):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)

                self._fit_one_step(state, action, reward, done, next_state)

                state = next_state
                total_reward += reward

                if done:
                    break

            totals_rewards.append(total_reward)

        return totals_rewards

env = gym.make('Acrobot-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = DQN(state_dim, action_dim)

episode_n = 100
t_max = 500
DQN_rewards = []
DQNH_rewards = []
DQNS_rewards = []
DDQN_rewards = []

agents ={'DQN':DQN(state_dim, action_dim),
         'DQNHardTargetUpdate':DQNHardTargetUpdate(state_dim, action_dim),
         'DQNSoftTargetUpdate':DQNSoftTargetUpdate(state_dim, action_dim),
         'DoubleDQN':DoubleDQN(state_dim, action_dim)}
full_rewards = {}
for agent_name, agent in agents.items():

    rewards = agent.fit(env, episode_n, t_max)
    full_rewards[agent_name] = rewards

plt.plot(full_rewards['DQN'])
plt.plot(full_rewards['DQNHardTargetUpdate'])
plt.plot(full_rewards['DQNSoftTargetUpdate'])
plt.plot(full_rewards['DoubleDQN'])
plt.legend( ['DQN','DQNH', 'DQNS', 'DDQN'])
plt.xlabel('Iterations')
plt.ylabel('Reward')
plt.savefig('Check_params2.jpg')
plt.show()