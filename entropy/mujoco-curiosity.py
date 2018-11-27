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
import collect

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
                    help='learning rate')
parser.add_argument('--lr', type=float, default=1e-4, metavar='lr',
                    help='learning rate')
parser.add_argument('--train_steps', type=int, default=2000, metavar='s',
                    help='number of steps per episodes')
parser.add_argument('--episodes', type=int, default=5000, metavar='e',
                    help='number of episodes per agent')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--collect', action='store_true',
                    help='collect a fresh set of policies')
parser.add_argument('--models_dir', type=str, default='models_cheetah/models_cheetah2018_11_19-07-42/', metavar='N',
                    help='directory from which to load model policies')
parser.add_argument('--learned_filename', type=str, default='learned/learned_policy.pt', metavar='ln',
                    help='file to save learned policy')
args = parser.parse_args()


def select_step(probs, var):
    # TODO: bug?
    m = Normal(probs.detach(), var)
    action = m.sample().numpy()[0]
    return action


def average_policies(policies):
    state_dict = policies[0].state_dict()
    for i in range(1, len(policies)):
        for k, v in policies[i].state_dict().items():
            state_dict[k] += v

    for k, v in state_dict.items():
        state_dict[k] /= float(len(policies))

    return state_dict

def grad_ent(pt):
    grad_p = -np.log(pt)
    grad_p[grad_p > 100] = 500
    return grad_p

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
    env = utils.env
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up experiment variables.
    iterations = 50
    T = 10000

    policies = load_from_dir(args.models_dir)
    # TODO: set obs_dim/action_dim/etc from the models if you loaded the models.

    # obtain average policy.
    average_policy_state_dict = average_policies(policies)
    exploration_policy = EntropyPolicy(utils.env, args.gamma, episodes=0, train_steps=0)
    exploration_policy.load_state_dict(average_policy_state_dict)
    average_p = exploration_policy.execute(T)
   
    print('*************')
    print(np.reshape(average_p, utils.space_dim))

    # Now, learn the actual reward structure based on environment rewards.
    actual_policy = ExplorePolicy(env, utils.obs_dim, utils.action_dim, exploration_policy, args.lr, args.gamma)
    actual_policy.learn_policy(args.episodes, args.train_steps)
    actual_policy.execute(T, render=True)
    actual_policy.save(learned_filename)

    env.close()

if __name__ == "__main__":
    main()






