import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions import Normal

from gym import wrappers
import utils

class CartEntropyPolicy(nn.Module):
    def __init__(self, env, gamma, obs_dim, action_dim):
        super(CartEntropyPolicy, self).__init__()

        # hidden_size = 128
        # self.affine1 = nn.Linear(obs_dim, hidden_size)
        # self.affine2 = nn.Linear(hidden_size, hidden_size) # TODO: get continuous action from this?

        # self.mean_head = nn.Linear(hidden_size, action_dim)
        # self.std_head = nn.Linear(hidden_size, action_dim)

        self.affine1 = nn.Linear(obs_dim, 128)
        # self.middle = nn.Linear(128, 128) # WORKING HERE
        self.affine2 = nn.Linear(128, action_dim)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2) # TODO(asoest): play with lr. try smaller.
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.init_state = np.array(self.env.env.state)

    # def forward(self, x):
    #     x_slim = x.narrow(1,0,self.obs_dim)
    #     x_slim = F.relu(self.affine1(x_slim))
    #     x_slim = F.relu(self.affine2(x_slim))

    #     mean = F.softmax(self.mean_head(x_slim), dim=1)
    #     std = F.softplus(self.std_head(x_slim))

    #     return mean, std

    # def get_probs(self, state):
    #     state = torch.from_numpy(state).float().unsqueeze(0)
    #     probs, var = self.forward(state)
    #     return probs, var

    # def select_action(self, state):
    #     state = torch.from_numpy(state).float().unsqueeze(0)
    #     probs, var = self.forward(state)

    #     # TODO: BUG? Is this right?
    #     m = Normal(probs.detach(), var)

    #     action = m.sample().numpy()[0]
    #     self.saved_log_probs.append(m.log_prob(action))
    #     return action

    def forward(self, x):
        x = F.relu(self.affine1(x))
        # x = F.relu(self.middle(x))
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
        # print(action)
        self.saved_log_probs.append(m.log_prob(action))
        if (action.item() == 1):
            return [0]
        elif (action.item() == 0):
            return [-1]
        return [1]
        # return [np.sign(action.item() - 0.5)] # WORKING HERE

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
            policy_loss.append(-log_prob * reward.float())

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.rewards.clear()
        self.saved_log_probs.clear()

    def get_obs(self):
        if utils.args.env == "Pendulum-v0":
            self.env.env.state = [np.pi, 0] # WORKING HERE
            return np.array(self.env.env._get_obs())
        elif utils.args.env == "MountainCarContinuous-v0":
            self.env.env.state = [-0.50, 0]
            return np.array(self.env.env.state)

    def learn_policy(self, reward_fn, episodes=1000, train_steps=1000):

        running_reward = 0
        for i_episode in range(episodes):
            # state = self.env.reset()
            self.env.reset()
            self.env.env.state = self.init_state
            state = self.get_obs()
            ep_reward = 0
            for t in range(train_steps):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, _, done, _ = self.env.step(action)
                reward = reward_fn[tuple(utils.discretize_state(state))]
                ep_reward += reward
                self.rewards.append(reward)
                if done:
                    # self.env.reset()
                    break

            running_reward = running_reward * 0.99 + ep_reward * 0.01
            if (i_episode == 0):
                running_reward = ep_reward
            
            self.update_policy()

            # Log to console.
            if i_episode % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))

    def execute(self, T, render=False, save_video_dir=''):
        p = np.zeros(shape=(tuple(utils.num_states)))
        # state = self.env.reset()

        if save_video_dir != '':
            self.env = wrappers.Monitor(self.env, save_video_dir)

        self.env.reset()
        self.env.env.state = self.init_state
        state = self.get_obs()
        for t in range(T):  
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            p[tuple(utils.discretize_state(state))] += 1

            if render:
                self.env.render()
                # time.sleep(.05)
            if done:
                break

        self.env.close()
        return p/float(T)

    def save(self, filename):
        torch.save(self, filename)