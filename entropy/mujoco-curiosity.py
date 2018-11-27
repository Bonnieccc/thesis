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
from torch.distributions import Normal

from entropy_policy import EntropyPolicy
from explore_policy import ExplorePolicy
import utils
from utils import args
import collect

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def select_step(probs, var):
    # TODO: bug?
    m = Normal(probs.detach(), var)
    action = m.sample().numpy()[0]
    return action


def load_from_dir(directory):
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
    np.set_printoptions(suppress=True, edgeitems=100)

    # Make environment.
    env = gym.make("HalfCheetah-v2")
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up experiment variables.
    iterations = 50
    T = 10000

    policies = load_from_dir(args.models_dir)
    # TODO: set obs_dim/action_dim/etc from the models if you loaded the models.

    # obtain average policy.
    average_policy_state_dict = collect.average_policies(policies)
    exploration_policy = EntropyPolicy(env, args.gamma, episodes=0, train_steps=0)
    exploration_policy.load_state_dict(average_policy_state_dict)
    average_p = exploration_policy.execute(T)
   
    print('*************')
    print(np.reshape(average_p, utils.space_dim))

    # Now, learn the actual reward structure based on environment rewards.
    actual_policy = ExplorePolicy(env, utils.obs_dim, utils.action_dim, exploration_policy, args.lr, args.gamma, args.eps)
    actual_policy.learn_policy(args.episodes, args.train_steps)
    actual_policy.execute(T, render=True)
    actual_policy.save(learned_filename)

    env.close()

if __name__ == "__main__":
    main()






