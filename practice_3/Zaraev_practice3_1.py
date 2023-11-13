from Frozen_Lake import FrozenLakeEnv
import numpy as np
import time
import operator
import matplotlib.pyplot as plt

def get_q_values(v_values, gamma):
    q_values = {}
    for state in env.get_all_states():
        q_values[state] = {}
        for action in env.get_possible_actions(state):
            q_values[state][action] = 0
            for next_state in env.get_next_states(state, action):
                q_values[state][action] += env.get_transition_prob(state, action, next_state) * env.get_reward(state, action, next_state)
                q_values[state][action] += gamma * env.get_transition_prob(state, action, next_state) * v_values[next_state]
    return q_values

def init_policy():
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        for action in env.get_possible_actions(state):
            policy[state][action] = 1 / len(env.get_possible_actions(state))
    return policy

def init_v_values():
    v_values = {}
    for state in env.get_all_states():
        v_values[state] = 0
    return v_values

def policy_evaluation_step(v_values, policy, gamma):
    q_values = get_q_values(v_values, gamma)
    new_v_values = init_v_values()
    for state in env.get_all_states():
        new_v_values[state] = 0
        for action in env.get_possible_actions(state):
            new_v_values[state] += policy[state][action] * q_values[state][action]
    return new_v_values

def policy_evaluation(policy, gamma, eval_iter_n):
    v_values = init_v_values()
    for _ in range(eval_iter_n):
        v_values = policy_evaluation_step(v_values, policy, gamma)
    q_values = get_q_values(v_values, gamma)
    return q_values

def policy_improvement(q_values):
    policy = {}
    for state in env.get_all_states():
        policy[state] = {}
        argmax_action = None
        max_q_value = float('-inf')
        for action in env.get_possible_actions(state):
            policy[state][action] = 0
            if q_values[state][action] > max_q_value:
                argmax_action = action
                max_q_value = q_values[state][action]
        policy[state][argmax_action] = 1
    return policy



iter_n = 100
eval_iter_n = 100
gamma_range = np.linspace(0, 1, 21)
results = {}


env = FrozenLakeEnv()

# policy = init_policy()
for gamma in gamma_range:
    policy = init_policy()
    for _ in range(iter_n):
        q_values = policy_evaluation(policy, gamma, eval_iter_n)
        policy = policy_improvement(q_values)

    total_rewards = []

    for _ in range(1000):
        total_reward = 0
        state = env.reset()
        for _ in range(1000):
            action = np.random.choice(env.get_possible_actions(state), p=list(policy[state].values()))
            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        total_rewards.append(total_reward)

    results[gamma] = np.mean(total_rewards)


plt.plot(list(results.keys()), list(results.values()))
plt.legend(['Cross Entropy'])
plt.xlabel('Iterations')
plt.title(f'gamma = {max(results.items(), key=operator.itemgetter(1))[0]}, '
          f'value = {max(results.items(), key=operator.itemgetter(1))[1]}')
plt.ylabel('Reward')
plt.savefig('Check_gamma.jpg')
plt.show()