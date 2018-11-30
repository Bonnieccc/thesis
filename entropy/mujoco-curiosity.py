# experimenting with curiosity exploration method.
# Code derived from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

# example command setting args in utils.py:
# python mujoco-curiosity.py  --models_dir=models-HalfCheetah-v2/models_2018_11_28-10-53/ --env="HalfCheetah-v2"

import os
import time

import numpy as np
import scipy.stats
import gym
from gym.spaces import prng

import torch
from torch.distributions import Normal

from cheetah_entropy_policy import CheetahEntropyPolicy
from explore_policy import ExplorePolicy
import utils
import collect

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
args = utils.get_args()


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
        if (file == "metadata"):
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
    T = 10000
    avg_runs = 10

    policies = load_from_dir(args.models_dir)
    
    for t in range(1, len(policies)):
        avg_state_dict = collect.average_policies(policies[:t])
        exploration_policy = CheetahEntropyPolicy(env, args.gamma)
        exploration_policy.load_state_dict(avg_state_dict)

        average_p = exploration_policy.execute(T)
        for i in range(avg_runs):
            average_p += exploration_policy.execute(T)
        average_p /= float(avg_runs)
        ent_average_p = scipy.stats.entropy(average_p.flatten())

        print('---------------------')
        print("Average policies[:%d]" % t)
        # print(average_p)
        print(ent_average_p)


    # obtain average policy.
    average_policy_state_dict = collect.average_policies(policies)
    exploration_policy = CheetahEntropyPolicy(env, args.gamma)
    exploration_policy.load_state_dict(average_policy_state_dict)
    average_p = exploration_policy.execute(T)
   
    print('*************')
    print(np.reshape(average_p, utils.space_dim))

    # Now, learn the actual reward structure based on environment rewards.
    # actual_policy = ExplorePolicy(env, utils.obs_dim, utils.action_dim, exploration_policy, args.lr, args.gamma, args.eps)
    # actual_policy.learn_policy(args.episodes, args.train_steps)
    # actual_policy.execute(T, render=True)
    # actual_policy.save()

    env.close()

if __name__ == "__main__":
    main()






