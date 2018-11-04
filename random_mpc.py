import os
import gym
import numpy as np
import time
import random
import math

def log_results(reward, steps, success):
    print("---------")
    print("Final reward: " + str(reward))
    print("Steps taken: " + str(steps))
    print("Success? " + str(success))
    print("---------")

def rollout_trajectory_weighted(trajectory, env, k, render=False, sleep=.10):
	# run the trajectory on the agent to see how long it lasts.
    max_x = -1.2
    total_reward = 0
    steps = 0
    state = env.reset()
    for t in range(k):
        steps += 1
        if render:
            env.render()
            time.sleep(sleep)
        obs, reward, done, info = env.step(trajectory[t])
        if obs[0] > max_x:
            max_x = obs[0]
        total_reward += reward
        if done:
            print("Episode: lasted {} timesteps, data: {}".format(t+1, obs))
            break
            
    # For continuous cartpole, return the total reward weighted 
    # by how close to the goal flag it got.
    return total_reward - total_reward*max_x, steps, steps<k

def rollout_trajectory(trajectory, env, k, render=False, sleep=.10):
    # run the trajectory on the agent to see how long it lasts.
    total_reward = 0
    steps = 0
    state = env.reset()
    for t in range(k):
        steps += 1
        if render:
            env.render()
            time.sleep(sleep)
        obs, reward, done, info = env.step(trajectory[t])
        total_reward += reward
        if done:
            print("Episode: lasted {} timesteps, data: {}".format(t+1, obs))
            break
    return total_reward, steps, steps<k

def control_random_mpc_pendulum(env, M, k, render=False):
    trajectories = []
    values = []
    # randomly select k actions for each of the M random trajectories
    for m in range(M):
        trajectory = []
        for t in range(k):
            action = env.action_space.sample()
            trajectory.append(action)

        trajectories.append(trajectory)
        value,_,_ = rollout_trajectory(trajectory, env, k)
        values.append(value)
 
    avg_trajectory = []
    for t in range(k):
        avg_value = 0
        for m in range(M):
            avg_value += values[m]*trajectories[m][t]

        step = avg_value
        avg_trajectory.append(step)

    reward,steps,success = rollout_trajectory(avg_trajectory, env, k, render=render, sleep=0.05)
    log_results(reward, steps, success)
    return reward, steps, success


def control_random_mpc_mountaincar_continuous(env, M, k, render=False):
    trajectories = []
    values = []
    # randomly select k actions for each of the M random trajectories
    for m in range(M):
        trajectory = []
        for t in range(k):
            action = env.action_space.sample()
            trajectory.append(action)

        trajectories.append(trajectory)
        weighted_value, steps, success = rollout_trajectory_weighted(trajectory, env, k)
        values.append(weighted_value)
 
    avg_trajectory = []
    for t in range(k):
        avg_value = 0
        for m in range(M):
            avg_value += values[m]*trajectories[m][t]

        step = avg_value
        avg_trajectory.append(step)

    weighted_reward, steps, success = rollout_trajectory_weighted(avg_trajectory, env, k, render=render, sleep=0.05)
    log_results(weighted_reward, steps, success)
    return weighted_reward, steps, success

def control_random_mpc_mountaincar_discrete(env, M, k, render=False):
    trajectories = []
    values = []
    # randomly select k actions for each of the M random trajectories
    for m in range(M):
        trajectory = []
        for t in range(k):
            action = env.action_space.sample()
            trajectory.append(action)

        trajectories.append(trajectory)
        value,_,_ = rollout_trajectory_weighted(trajectory, env, k)
        values.append(value)

    # Normalize the losses vector
    values = [float(i)/sum(values) for i in values]
 
    avg_trajectory = []
    for t in range(k):
        avg_value = {0:0, 1:0, 2:0}
        for m in range(M):
            avg_value[trajectories[m][t]] += values[m]

        # pick action with max weight
        step = max(avg_value, key=avg_value.get)
        avg_trajectory.append(step)

    reward, steps, success = rollout_trajectory_weighted(avg_trajectory, env, k, render=render, sleep=0.05)
    log_results(reward, steps, success)
    return reward, steps, success

def control_random_mpc_cartpole(env, M, k, render=False):
    trajectories = []
    values = []
    # randomly select k actions for each of the M random trajectories
    for m in range(M):
        trajectory = []
        for t in range(k):
            action = env.action_space.sample()
            trajectory.append(action)

        trajectories.append(trajectory)
        value,_,_ = rollout_trajectory(trajectory, env, k)
        values.append(value)

    # Normalize the losses vector
    values = [float(i)/sum(values) for i in values]

    avg_trajectory = []
    for t in range(k):
        avg_value = {0:0, 1:0}
        for m in range(M):
            avg_value[trajectories[m][t]] += values[m]

        # pick action with weighted probability
        # step = np.random.choice(2, 1, p=[avg_value[0], avg_value[1]])[0]

        # pick action with max weight
        step = max(avg_value, key=avg_value.get)
        avg_trajectory.append(step)

    reward, steps, failure = rollout_trajectory(avg_trajectory, env, k, render=render, sleep=.05)
    log_results(reward, steps, not failure)
    return reward, steps, not failure
