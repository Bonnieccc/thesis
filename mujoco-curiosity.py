# experimenting with curiosity exploration method.
# Code derived from: https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py

import os
import sys
import time
from datetime import datetime
import logging
import argparse

import numpy as np
from itertools import count

import gym
from gym.spaces import prng

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

import mujoco_py

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# TODO:
# need to limit the bounds on pbservation space (currently +-inf)
# for anything above X, set val = X
# discretize state space along 111 axis


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
                    help='learning rate')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--collect', action='store_true',
                    help='collect a fresh set of policies')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--models_dir', type=str, default='models_cheetah/models_cheetah2018_11_17-13-49/', metavar='N',
                    help='directory from which to load model policies')
args = parser.parse_args()


def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

# Set up environment.

env = gym.make("HalfCheetah-v2")
num_bins = 10
obs_dim = 3 #len(env.observation_space.high)
action_dim = 6
state_bins = []
for i in range(obs_dim):
    state_bins.append(discretize_range(-2, 2, num_bins))

num_states = []
for i in range(obs_dim):
    num_states.append(len(state_bins[i]) + 1)

def discretize_state(observation):
    # Discretize the observation features and reduce them to a single list.
    state = []
    for i, feature in enumerate(observation):
        if i>=obs_dim:
            break
        state.append(discretize_value(feature, state_bins[i]))
    return state


class Policy(nn.Module):
    def __init__(self):

        super(Policy, self).__init__()
        self.affine1 = nn.Linear(obs_dim, 128)
        self.affine2 = nn.Linear(128, 128) # TODO: get continuous action from this?

        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.std_head = nn.Linear(hidden_size, action_dim)

        self.action_scale = 2.0

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3) # TODO(asoest): play with lr. try smaller.
        self.eps = np.finfo(np.float32).eps.item()


    def forward(self, x):
        x_slim = x.narrow(1,0,obs_dim)
        x_slim = F.relu(self.affine1(x_slim))
        x_slim = F.relu(self.affine2(x_slim))

        mean = F.softmax(self.mean_head(x_slim), dim=1)
        std = F.softplus(self.std_head(x_slim))

        return mean, std

    def get_probs(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, var = self.forward(state)
        return probs, var

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, var = self.forward(state)

        # TODO: BUG? Is this right?
        # m = Normal(torch.tensor([0., 0., 0., 0., 0., 0.]), probs)
        m = Normal(probs.detach(), var)

        action = m.sample().numpy()[0]
        self.saved_log_probs.append(m.log_prob(action))
        return action

    def update_policy(self):
        R = 0
        policy_loss = []
        rewards = []

        # Get discounted rewards from the episode.
        for r in self.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.rewards.clear()
        self.saved_log_probs.clear()

    def learn_policy(self, reward_fn):

        running_reward = 0
        for i_episode in range(100):
            state = env.reset()
            ep_reward = 0
            for t in range(1000):  # Don't infinite loop while learning
                action = self.select_action(state)
                # print(action)
                state, _, done, _ = env.step(action)
                # print(state)
                reward = reward_fn[tuple(discretize_state(state))]
                ep_reward += reward
                self.rewards.append(reward)
                if done:
                    env.reset()

            running_reward = running_reward * 0.99 + ep_reward * 0.01
            if (i_episode == 0):
                running_reward = ep_reward
            
            self.update_policy()

            # Log to console.
            if i_episode % args.log_interval == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))

    def execute(self, T):
        p = np.zeros(shape=(tuple(num_states)))
        state = env.reset()
        for t in range(T):  
            action = self.select_action(state)
            state, reward, done, _ = env.step(action)
            p[tuple(discretize_state(state))] += 1

            if done:
                env.reset()

        return p/float(T)

def select_step(probs, var):
    # TODO: bug?
    # m = Normal(torch.tensor([0., 0., 0., 0., 0., 0.]), probs)
    m = Normal(probs.detach(), var)
    action = m.sample().numpy()[0]
    return action

def execute_average_policy(policies, T):
    # run a simulation to see how the average policy behaves.
    p = np.zeros(shape=(tuple(num_states)))
    state = env.reset()
    for i in range(T):
        # Compute average probability over action space for state.
        probs = torch.tensor(np.zeros(shape=(1,6))).float()
        var = torch.tensor(np.zeros(shape=(1,6))).float()
        for policy in policies:
            p, v = policy.get_probs(state)
            probs += p
            var += v
        probs /= len(policies)
        var /= len(policies)

        # Select a step.
        action = select_step(probs, var)
        state, reward, done, _ = env.step(action)
        p[tuple(discretize_state(state))] += 1

        if args.render:
            env.render()
        if done:
            env.reset()

    return p / float(T)

# TODO: Is this right????
def grad_ent(pt):
    grad_p = -np.log(pt)
    grad_p[grad_p > 100] = 500
    return grad_p

def log_iteration(i, logger, p, reward_fn):
    np.set_printoptions(suppress=True, threshold=num_bins**obs_dim)

    if i == 'average':
        logger.debug("*************************")

    logger.debug('Iteration: ' + str(i))
    logger.debug('p' + str(i) + ':')
    logger.debug(np.reshape(p, num_bins**obs_dim))
    logger.debug('reward_fn' + str(i) + ':')
    logger.debug(np.reshape(reward_fn, num_bins**obs_dim))

    np.set_printoptions(suppress=True, threshold=100, edgeitems=100)

# Iteratively collect and learn T policies using policy gradients and a reward
# function based on entropy.
def collect_entropy_policies(iterations, T, current_time, logger):
    MODEL_DIR = 'models_cheetah/models_cheetah' + current_time + '/'
    os.mkdir(MODEL_DIR)
    reward_fn = np.ones(shape=(tuple(num_states)))
    policies = []
    for i in range(iterations):
        # Learn policy that maximizes current reward function.
        policy = Policy()
        policy.learn_policy(reward_fn)

        # Get next distribution p by executing pi for T steps.
        p = policy.execute(T)

        log_iteration(i, logger, p, reward_fn)

        print("p=")
        print(np.reshape(p, num_bins**obs_dim))
        print("reward_fn=")
        print(np.reshape(reward_fn, num_bins**obs_dim))
        print("----------------------")

        reward_fn = grad_ent(p)
        policies.append(policy)

        # save the policy
        torch.save(policy, MODEL_DIR + 'model_' + str(i) + '.pt')

    return policies

def load_from_dir(directory):
    # for all
    policies = []
    files = os.listdir(directory)
    for file in files:
        policy = torch.load(directory + file)
        policies.append(policy)
    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)

    # Make environment.
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up experiment variables.
    iterations = 20
    T = 10000

    # Collect policies.
    if args.collect:
        # set up logging to file 
        TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
        LOG_DIR = 'logs-cheetah/'
        FILE_NAME = 'test' + TIME
        logging.basicConfig(level=logging.DEBUG,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=LOG_DIR + FILE_NAME + '.log',
                            filemode='w')
        logger = logging.getLogger('cheetah-curiosity.pt')
        policies = collect_entropy_policies(iterations, T, TIME, logger)
    else:
        policies = load_from_dir(args.models_dir)
   
    average_p = execute_average_policy(policies, T)

    if args.collect:
        log_iteration('average', logger, average_p, [])

    print('*************')
    print(np.reshape(average_p, num_bins**obs_dim))

    env.close()

    # display histogram of data.
    # # You can set arbitrary bin edges:
    # bins = [0, 0.150, .30, .45, .60]
    # hist, bin_edges = np.histogram(a, bins=bins)
    bins = [0, .00009, .00019, .00029, .00039, .00049, .00059, .00069, .00079]
    plt.hist(np.reshape(average_p, num_bins**obs_dim), bins=bins)
    plt.show()
            

if __name__ == "__main__":
    main()





