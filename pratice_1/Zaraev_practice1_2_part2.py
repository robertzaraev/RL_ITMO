import gym
import numpy as np
import random
import time
import matplotlib.pyplot as plt

env = gym.make("Taxi-v3")
env.reset(return_info=True)
print("reset: ", env.reset(return_info=True)[0])
ACTION_N = 6
STATE_N = 500


class RandomAgent():
    def __init__(self, action_n):
        self.action_n = action_n

    def get_action(self, state):
        action = np.random.randint(self.action_n)
        return action

class CrossEntropyAgent():
    def __init__(self, state_n, action_n):
        self.laplase = 5
        self.state_n = state_n
        self.action_n = action_n
        self.model = np.ones((self.state_n, self.action_n)) / self.action_n

    def get_action(self, state):
        # print(np.sum(self.model[state]))
        action = np.random.choice(np.arange(self.action_n), p=self.model[state])
        return int(action)

    def fit(self, elite_trajectories):
        new_model = np.zeros((self.state_n, self.action_n))

        for trajectory in elite_trajectories:
            for state, action in zip(trajectory['states'], trajectory['actions']):
                new_model[state][action] += 1 + self.laplase

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


# agent = RandomAgent(ACTION_N)

agent = CrossEntropyAgent(state_n=STATE_N,
                          action_n=ACTION_N)
q_param = 0.6
iteration_n = 10
trajectory_n = 1500
l_value = .95
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
    elite_trajectories_indexes = []
    not_elite_trajectories_indexes = []
    for index, trajectory in enumerate(trajectories):
        total_reward = np.sum(trajectory['rewards'])

        if total_reward > quantile:
            if random.random() <= l_value:
                elite_trajectories.append(trajectory)
        else:
            if random.random() > l_value:
                elite_trajectories.append(trajectory)

    agent.fit(elite_trajectories)

plt.plot(range(iteration_n), mean_total_rewards)
plt.legend(['Cross Entropy'])
plt.xlabel('Iterations')
plt.title(f'q_param - {q_param} iteration_n : {iteration_n} trajectory_n : {trajectory_n} max_reward : {max(mean_total_rewards)}')
plt.ylabel('Reward')
plt.savefig('Policy_best.jpg')
plt.show()


