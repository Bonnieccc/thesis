# Original code author: Yuriy Guts
# Github link: https://github.com/YuriyGuts/cartpole-q-learning/blob/master/cartpole.py

import os
import gym
import numpy as np
import time
import random
import math


def rollout_trajectory(trajectory, env, k, render=False, sleep=.10):
    # run the trajectory on the agent to see how long it lasts.
    lasted = 0
    total_reward = 0
    state = env.reset()
    for t in range(k):
        if render:
            env.render()
            time.sleep(sleep)
        obs, reward, done, info = env.step(trajectory[t])
        total_reward += reward
        if done:
            print("Episode: lasted {} timesteps, data: {}".format(t+1, obs))
            break
    return total_reward

def control_random_mpc_mountaincart_continuous(env, M, k):
    trajectories = []
    values = []
    # randomly select k actions for each of the M random trajectories
    for m in range(M):
        trajectory = []
        for t in range(k):
            action = env.action_space.sample()
            trajectory.append(action)

        trajectories.append(trajectory)
        value = rollout_trajectory(trajectory, env, k)
        values.append(value)

    print(values)
 
    avg_trajectory = []
    for t in range(k):
        avg_value = 0
        for m in range(M):
            avg_value += values[m]*trajectories[m][t]

        step = avg_value
        avg_trajectory.append(step)

    reward = rollout_trajectory(avg_trajectory, env, k, render=True, sleep=0.05)
    print("reward: " + str(reward))

def control_random_mpc_mountaincart_discrete(env, M, k):
    trajectories = []
    values = []
    # randomly select k actions for each of the M random trajectories
    for m in range(M):
        trajectory = []
        for t in range(k):
            action = env.action_space.sample()
            trajectory.append(action)

        trajectories.append(trajectory)
        value = rollout_trajectory(trajectory, env, k)
        values.append(value)

    # Normalize the losses vector
    values = [float(i)/sum(values) for i in values]
    print(values)

 
    avg_trajectory = []
    for t in range(k):
        avg_value = {0:0, 1:0, 2:0}
        for m in range(M):
            avg_value[trajectories[m][t]] += values[m]

        step = np.random.choice(3, 1, p=[avg_value[0], avg_value[1], avg_value[2]])[0]
        avg_trajectory.append(step)

    reward = rollout_trajectory(avg_trajectory, env, k, render=True, sleep=0.05)
    print("reward: " + str(reward))

def control_random_mpc_cartpole(env, M, k):
    trajectories = []
    values = []
    # randomly select k actions for each of the M random trajectories
    for m in range(M):
        trajectory = []
        for t in range(k):
            action = env.action_space.sample()
            trajectory.append(action)

        trajectories.append(trajectory)
        value = rollout_trajectory(trajectory, env, k)
        values.append(value)

    # Normalize the losses vector
    values = [float(i)/sum(values) for i in values]
    print(values)

    avg_trajectory = []
    for t in range(k):
        avg_value = {0:0, 1:0}
        for m in range(M):
            avg_value[trajectories[m][t]] += values[m]

        step = np.random.choice(2, 1, p=[avg_value[0], avg_value[1]])[0]
        avg_trajectory.append(step)

    reward = rollout_trajectory(avg_trajectory, env, k)
    print("selected trajectory:" + str(avg_trajectory))
    print("reward: " + str(reward))

def main():
    monitor = False
    random.seed(time.time())
    np.random.seed(seed=int(time.time()))
    # random_state = np.random.randint(20)

    cartpole_env = gym.make("CartPole-v1")
    mountaincar_env = gym.make("MountainCar-v0")
    mountaincar_cont_env = gym.make("MountainCarContinuous-v0")
    
    cartpole_env.seed(int(time.time()))
    mountaincar_env.seed(int(time.time()))
    mountaincar_cont_env.seed(int(time.time()))

    # control_random_mpc_cartpole(cartpole_env, 100, 100)
    # control_random_mpc_mountaincart_discrete(mountaincar_env, 100, 100000)
    control_random_mpc_mountaincart_continuous(mountaincar_cont_env, 20, 1000000)

if __name__ == "__main__":
    main()

    # This trajectory lasted 100 timesteps:
    # [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1]

    # This one lasted 74
    # [1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1]

    # 92
    # [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1]