import numpy as np
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Vanilla Policy Gradient to learn entropy-based rewards.
class CheetahEntropyPolicy(nn.Module):
    def __init__(self, env, gamma): 

        super(CheetahEntropyPolicy, self).__init__()
        self.obs_dim = utils.obs_dim
        self.action_dim = utils.action_dim
        self.hidden_size = 128
        self.affine1 = nn.Linear(self.obs_dim, self.hidden_size)
        self.affine2 = nn.Linear(self.hidden_size, self.hidden_size) # TODO: get continuous action from this?

        self.mean_head = nn.Linear(self.hidden_size, self.action_dim)
        self.std_head = nn.Linear(self.hidden_size, self.action_dim)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=1e-4) # TODO(asoest): play with lr. try smaller.
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env
        self.gamma = gamma

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

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, var = self.forward(state)

        # TODO: BUG? Is this right?
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

    def learn_policy(self, reward_fn, episodes, train_steps):

        running_reward = 0
        for i_episode in range(episodes):
            state = self.env.reset()
            ep_reward = 0
            for t in range(train_steps):
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

    def execute(self, T, render=False):
        p = np.zeros(shape=(tuple(utils.num_states)))
        state = self.env.reset()
        for t in range(T):  
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            p[tuple(utils.discretize_state(state))] += 1

            if render:
                self.env.render()
            if done:
                self.env.reset()

        return p/float(T)


    def save(self, filename):
        torch.save(self, filename)
