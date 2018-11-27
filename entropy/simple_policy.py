import numpy as np
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

# Set up environment.
nx = 10
nv = 4
obs_dim = 2
action_dim = 2
# S = nx*nv

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

class SimplePolicy(nn.Module):
    def __init__(self, env, gamma=.99):
        super(SimplePolicy, self).__init__()
        self.affine1 = nn.Linear(2, 128)
        self.affine2 = nn.Linear(128, 2)

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
                reward = reward_fn[tuple(discretize_state(state))]
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
        p = np.zeros(shape=(num_states[0],
                           num_states[1]))
        state = self.env.reset()
        for t in range(T):  
            action = self.select_action(state)
            state, reward, done, _ = self.env.step(action)
            p[tuple(discretize_state(state))] += 1

            if done:
                self.env.reset()

        return p/float(T)

    def save(self, filename):
        torch.save(self, filename)