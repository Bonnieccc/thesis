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
from torch.distributions import Categorical


parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
                    help='learning rate')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--collect', action='store_true',
                    help='collect a fresh set of policies')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--models_dir', type=str, default='models_2018_11_11-23-15/', metavar='N',
                    help='directory from which to load model policies')
args = parser.parse_args()



def discretize_range(lower_bound, upper_bound, num_bins):
	return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
	return np.asscalar(np.digitize(x=value, bins=bins))

# Set up environment.
nx = 10
nv = 4
S = nx*nv

state_bins = [
    # Cart position.
    discretize_range(-1.2, 0.6, nx), 
    # Cart velocity.
    discretize_range(-0.07, 0.07, nv)
]

num_states = [(len(state_bins[0]) + 1),
              (len(state_bins[1]) + 1)]


def discretize_state(observation):
    # Discretize the observation features and reduce them to a single list.
    state = []
    for i, feature in enumerate(observation):
        state.append(discretize_value(feature, state_bins[i]))
    return state


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(2, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2)
        self.eps = np.finfo(np.float32).eps.item()

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

    def get_probs(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        return probs

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item() - 0.5

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

    def learn_policy(self, reward_fn, env):

        running_reward = 0
        for i_episode in range(300):
            state = env.reset()
            ep_reward = 0
            for t in range(1000):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, _, done, _ = env.step([action])
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

    def execute(self, env, T):
        p = np.zeros(shape=(num_states[0],
                           num_states[1]))
        state = env.reset()
        for t in range(T):  
            action = self.select_action(state)
            state, reward, done, _ = env.step([action])
            p[tuple(discretize_state(state))] += 1

            if done:
                env.reset()

        return p/float(T)

def select_step(probs):
    m = Categorical(probs)
    action = m.sample()
    return action.item() - 0.5

def execute_average_policy(env, policies, T):
    # run a simulation to see how the average policy behaves.
    p = np.zeros(shape=(num_states[0],
                        num_states[1]))
    state = env.reset()
    for i in range(T):
        # Compute average probability over action space for state.
        probs = torch.tensor([[0., 0.]])
        for policy in policies:
            probs += policy.get_probs(state)
        probs /= len(policies)

        # Select a step.
        action = select_step(probs)
        state, reward, done, _ = env.step([action])
        p[tuple(discretize_state(state))] += 1

        if args.render:
            env.render()
        if done:
            env.reset()

    return p / float(T)

# TODO: Is this right????
def grad_ent(pt):
    grad_p = -np.log(pt)
    grad_p[grad_p > 100] = 200
    return grad_p

# Iteratively collect and learn T policies using policy gradients and a reward
# function based on entropy.
def collect_entropy_policies(env, iterations, T, logger):
    MODEL_DIR = 'models_' + TIME + '/'
    os.mkdir(MODEL_DIR)
    reward_fn = np.ones(shape=(num_states[0],
                               num_states[1]))
    policies = []
    for i in range(iterations):
        # Learn policy that maximizes current reward function.
        policy = Policy()
        policy.learn_policy(reward_fn, env)

        # Get next distribution p by executing pi for T steps.
        p = policy.execute(env, T)

        logger.debug('Iteration: ' + str(i))
        logger.debug(p)
        logger.debug(reward_fn)

        print("p=")
        print(p)
        print("reward_fn=")
        print(reward_fn)
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

    np.set_printoptions(suppress=True)

    # Make environment.
    env = gym.make("MountainCarContinuous-v0")
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up experiment variables.
    iterations = 100
    T = 10000

    # Collect policies.
    if args.collect:
        # set up logging to file 
        TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
        LOG_DIR = 'logs/'
        FILE_NAME = 'test' + TIME
        logging.basicConfig(level=logging.DEBUG,
                            format='%(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=LOG_DIR + FILE_NAME + '.log',
                            filemode='w')
        logger = logging.getLogger('curiosity.pt')
        policies = collect_entropy_policies(env, iterations, T, logger)
    else:
        policies = load_from_dir(args.models_dir)
   
    average_p = execute_average_policy(env, policies, T)


    if args.collect:
        logger.debug('*************')
        logger.debug(average_p)

    print('*************')
    print(average_p)

    env.close()
        

if __name__ == "__main__":
    main()





