# experimenting with curiosity exploration method.
# Code derived from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

# example command setting args in utils.py
# python curiosity.py  --models_dir=models-MountainCarContinuous-v0/models_2018_11_28-17-45/ --env="MountainCarContinuous-v0"
# python curiosity.py  --models_dir=models-Pendulum-v0/models_2018_11_29-09-48/ --env="Pendulum-v0"

import os
import sys
import time

import random
import numpy as np
import scipy.stats
import gym
from gym.spaces import prng

import torch
from torch.distributions import Categorical

from explore_policy import ExplorePolicy
from cart_entropy_policy import CartEntropyPolicy
import utils
import collect

args = utils.get_args()

def select_action(probs):
    m = Categorical(probs)
    action = m.sample()
    if (action.item() == 1):
        return [0]
    elif (action.item() == 0):
        return [-1]
    return [1]

def load_from_dir(directory):
    policies = []
    files = os.listdir(directory)
    for file in sorted(files):
        if (file == "metadata"):
            print("skipping: " + file)
            continue
        policy = torch.load(directory + file)
        policies.append(policy)
    return policies

def get_next_file(directory, model_time, ext, dot=".png"):
    i = 0
    fname = directory + model_time + ext
    while os.path.isfile(fname):
        fname = directory + model_time + ext + str(i) + dot
        i += 1
    return fname

def execute_average_policy(env, policies, T, avg_runs=1, render=False):
    # run a simulation to see how the average policy behaves.

    random_T = np.floor(random.random()*T)
    
    average_p = np.zeros(shape=(tuple(utils.num_states)))
    avg_entropy = 0
    random_initial_state = []

    for i in range(avg_runs):
        # unroll for T steps and compute p
        p = np.zeros(shape=(tuple(utils.num_states)))
        state = env.reset()
        for i in range(T):
            # Compute average probability over action space for state.
            probs = torch.tensor(np.zeros(shape=(1,utils.action_dim))).float()
            var = torch.tensor(np.zeros(shape=(1,utils.action_dim))).float()
            for policy in policies:
                prob = policy.get_probs(state)
                probs += prob
            probs /= len(policies)
            action = select_action(probs)
            
            state, reward, done, _ = env.step(action)
            p[tuple(utils.discretize_state(state))] += 1
            if (i == random_T):
                random_initial_state = state
                print(random_initial_state)

            if render and i % 10 == 0:
                env.render()
                time.sleep(.05)
            if done:
                env.reset()

        p /= float(T)
        average_p += p
        avg_entropy += scipy.stats.entropy(average_p.flatten())

    env.close()
    average_p /= float(avg_runs)

    avg_entropy /= float(avg_runs) # running average of the entropy 
    entropy_of_final = scipy.stats.entropy(average_p.flatten())

    # print("compare:")
    # print(avg_entropy) # running entropy
    # print(entropy_of_final) # entropy of the average distribution

    return average_p, avg_entropy, random_initial_state


def average_p_and_entropy(env, policies, T, avg_runs=1, render=False, save_video_dir=''):
    exploration_policy = collect.average_policies(env, policies)
    average_p = exploration_policy.execute(T, render=render, save_video_dir=save_video_dir)
    average_ent = scipy.stats.entropy(average_p.flatten())
    for i in range(avg_runs-1):
        p = exploration_policy.execute(T)
        average_p += p
        average_ent += scipy.stats.entropy(p.flatten())
        # print(scipy.stats.entropy(p.flatten()))
    average_p /= float(avg_runs) 
    average_ent /= float(avg_runs) # 
    entropy_of_final = scipy.stats.entropy(average_p.flatten())

    # print("entropy compare: ")
    # print(average_ent) # running entropy
    # print(entropy_of_final) # entropy of final

    # return average_p, avg_entropy
    return average_p, average_ent

def main():
    # Suppress scientific notation.
    np.set_printoptions(suppress=True)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up experiment variables.
    T = 1000
    avg_runs = 10

    policies = load_from_dir(args.models_dir)
    
    times = []
    entropies = []

    x_dist_times = []
    x_distributions = []

    v_dist_times = []
    v_distributions = []
    
    for t in range(1, len(policies)):

        average_p, avg_entropy = average_p_and_entropy(policies[:t], avg_runs)
        
        print('---------------------')
        print("Average policies[:%d]" % t)
        print(average_p)
        print(avg_entropy)

    # obtain global average policy.
    exploration_policy = collect.average_policies(env, policies)
    average_p = exploration_policy.execute(T)
   
    print('*************')
    print(average_p)

    # actual_policy = ExplorePolicy(env, obs_dim, action_dim, exploration_policy, args.lr, args.gamma)
    # actual_policy.learn_policy(args.episodes, args.train_steps)
    # actual_policy.execute(T, render=True)
    # actual_policy.save()

    env.close()
        

if __name__ == "__main__":
    main()





