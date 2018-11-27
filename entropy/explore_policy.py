import numpy as np
import gym
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Vanilla Policy Gradient with entropy exploration.
class ExplorePolicy(nn.Module):
    def __init__(self, env, obs_dim, action_dim, exploration_policy, lr, gamma, eps=0.05): 

        super(ExplorePolicy, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_size = 128
        self.affine1 = nn.Linear(self.obs_dim, self.hidden_size)
        self.affine2 = nn.Linear(self.hidden_size, self.hidden_size) # TODO: get continuous action from this?

        self.mean_head = nn.Linear(self.hidden_size, self.action_dim)
        self.std_head = nn.Linear(self.hidden_size, self.action_dim)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=lr) # TODO: play with lr
        self.eps = np.finfo(np.float32).eps.item()

        self.exploration_policy = exploration_policy
        self.env = env
        self.epsilon = eps # TODO: play with this. 
        self.gamma = gamma

        random.seed(time.time())


    def forward(self, x):
        x_slim = x.narrow(1,0,self.obs_dim)
        x_slim = F.relu(self.affine1(x_slim))
        x_slim = F.relu(self.affine2(x_slim))

        mean = F.softmax(self.mean_head(x_slim), dim=1)
        std = F.softplus(self.std_head(x_slim))

        return mean, std

    def get_probs(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, var = self.forward(state)
        return probs, var

    def get_distributions(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, var = self.forward(state)
        m = Normal(probs.detach(), var) 
        return m

    def save_log_prob(self, state, action):
        m = self.get_distributions(state)
        self.saved_log_probs.append(m.log_prob(action))

    def select_action(self, state):
        m = self.get_distributions(state)
        action = m.sample().numpy()[0]
        return action

    def update_policy(self):
        R = 0
        policy_loss = []
        rewards = []

        # Get discounted rewards from the episode.
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.eps)

        for log_prob, reward in zip(self.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        # Reset for next batch.
        self.rewards.clear()
        self.saved_log_probs.clear()

    def run_exploration(self, train_steps):
        print("Exploring!")
        state = self.env.reset()
        ep_reward = 0
        for t in range(train_steps):

            action = self.exploration_policy.select_action(state)
            self.save_log_prob(state, action[0])
            state, reward, done, _ = self.env.step(action)

            ep_reward += reward
            
            self.rewards.append(reward)
            if done:
                self.env.reset()

        return ep_reward

    def run_normal(self, train_steps):
        state = self.env.reset()
        ep_reward = 0
        for t in range(train_steps):
            action = self.select_action(state)
            self.save_log_prob(state, action)
            state, reward, done, _ = self.env.step(action)
            ep_reward += reward
            
            self.rewards.append(reward)
            if done:
                self.env.reset()

        return ep_reward

    def learn_policy(self, episodes, train_steps):

        running_reward = 0
        best_reward = np.NINF
        for i_episode in range(episodes):

            # ep_reward = 0
            if (random.random() < self.epsilon):
                ep_reward = self.run_exploration(train_steps)
            else:
                ep_reward = self.run_normal(train_steps)

            if ep_reward > best_reward:
                best_reward = ep_reward

            running_reward = running_reward * 0.99 + ep_reward * 0.01
            if (i_episode == 0):
                running_reward = ep_reward
            
            self.update_policy()

            # Log to console.
            if i_episode % 10 == 0:
                print('Episode {}\tLast: {:.2f}\tAverage: {:.2f}\tBest: {:.2f}'.format(
                    i_episode, ep_reward, running_reward, best_reward))

    def execute(self, T, render=False):
        state = self.env.reset()
        for t in range(T):  
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)

            if render:
                self.env.render()
            if done:
                self.env.reset()

        self.env.close()

    def save(self, filename):
        torch.save(self, filename)
