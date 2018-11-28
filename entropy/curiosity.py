# experimenting with curiosity exploration method.
# Code derived from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import os
import sys
import time
from datetime import datetime
import logging
import argparse

import numpy as np
import gym
from gym.spaces import prng

import torch
from torch.distributions import Categorical

from explore_policy import ExplorePolicy
from simple_policy import SimplePolicy
import utils
from utils import args
import collect


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
        policy = torch.load(directory + file)
        policies.append(policy)
    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True)

    # Make environment.
    env = gym.make("MountainCarContinuous-v0")
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up experiment variables.
    T = 10000

    policies = load_from_dir(args.models_dir)

    # obtain average policy.
    average_policy_state_dict = collect.average_policies(policies)
    exploration_policy = SimplePolicy(env)
    exploration_policy.load_state_dict(average_policy_state_dict)
    average_p = exploration_policy.execute(env, T)
   
    print('*************')
    print(average_p)

    actual_policy = ExplorePolicy(env, obs_dim, action_dim, exploration_policy, args.lr, args.gamma)
    actual_policy.learn_policy(args.episodes, args.train_steps)
    actual_policy.execute(T, render=True)
    actual_policy.save()

    env.close()
        

if __name__ == "__main__":
    main()





