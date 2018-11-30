# experimenting with curiosity exploration method.
# Code derived from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

# example command setting args in utils.py
# python curiosity.py  --models_dir=models-MountainCarContinuous-v0/models_2018_11_28-17-45/ --env="MountainCarContinuous-v0"

import os
import sys
import time
from datetime import datetime
import json

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

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

args = utils.get_args()


def select_step(probs):
    m = Categorical(probs)
    action = m.sample()
    return action.item() - 0.5


def load_from_dir(directory):
    # for all
    policies = []
    files = os.listdir(directory)
    for file in files:
        if (file == "model_0.pt"):
            print("skipping: " + file)
            continue
        if (file == "metadata"):
            print("skipping: " + file)
            continue
        policy = torch.load(directory + file)
        policies.append(policy)
    return policies

def main():
    # Suppress scientific notation.
    np.set_printoptions(suppress=True)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up experiment variables.
    T = 10000
    avg_runs = 10

    policies = load_from_dir(args.models_dir)
    
    times = []
    entropies = []
    
    for t in range(1, len(policies)):
        avg_state_dict = collect.average_policies(policies[:t])
        exploration_policy = CartEntropyPolicy(env, args.gamma, utils.obs_dim, utils.action_dim)
        exploration_policy.load_state_dict(avg_state_dict)

        average_p = exploration_policy.execute(T)
        for i in range(avg_runs):
            average_p += exploration_policy.execute(T)
        average_p /= float(avg_runs)
        ent_average_p = scipy.stats.entropy(average_p.flatten())
        
        times.append(t)
        entropies.append(ent_average_p)

        print('---------------------')
        print("Average policies[:%d]" % t)
        print(average_p)
        print(ent_average_p)

    plt.plot(times, entropies)
    plt.show()
    plt.savefig(args.models_dir + "entropy.png")

    # obtain global average policy.
    average_policy_state_dict = collect.average_policies(policies)
    exploration_policy = CartEntropyPolicy(env, args.gamma, utils.obs_dim, utils.action_dim)
    exploration_policy.load_state_dict(average_policy_state_dict)
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





