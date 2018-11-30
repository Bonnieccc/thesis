import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import utils

class CartEntropyPolicy(nn.Module):
    def __init__(self, env, gamma, obs_dim, action_dim):
        super(CartEntropyPolicy, self).__init__()
        self.affine1 = nn.Linear(obs_dim, 128)
        self.affine2 = nn.Linear(128, action_dim)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=1e-2) # TODO(asoest): play with lr. try smaller.
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env
        self.gamma = gamma

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
        return [action.item() - 0.5]

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

    def learn_policy(self, reward_fn, episodes=1000, train_steps=1000):

        running_reward = 0
        for i_episode in range(episodes):
            state = self.env.reset()
            ep_reward = 0
            for t in range(train_steps):  # Don't infinite loop while learning
                action = self.select_action(state)
                state, _, done, _ = self.env.step(action)
                reward = reward_fn[tuple(utils.discretize_state(state))]
                ep_reward += reward
                self.rewards.append(reward)
                if done:
                    self.env.reset()

            running_reward = running_reward * 0.99 + ep_reward * 0.01
            if (i_episode == 0):
                running_reward = ep_reward
            
            self.update_policy()

            # Log to console.
            if i_episode % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))

    def execute(self, T):
        # p = np.zeros(shape=(utils.num_states[0],
        #                    utils.num_states[1]))
        p = np.zeros(shape=(tuple(utils.num_states)))
        state = self.env.reset()
        for t in range(T):  
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            p[tuple(utils.discretize_state(state))] += 1

            if done:
                self.env.reset()

        return p/float(T)

    def save(self, filename):
        torch.save(self, filename)