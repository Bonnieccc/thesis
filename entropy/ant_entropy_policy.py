import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from gym.spaces import prng
from gym import wrappers
import ant_utils

class AntEntropyPolicy(nn.Module):
    def __init__(self, env, gamma, lr, obs_dim, action_dim):
        super(AntEntropyPolicy, self).__init__()

        self.affine1 = nn.Linear(obs_dim, 128)
        self.middle = nn.Linear(128, 128)
        self.mu = nn.Linear(128, action_dim)
        self.sigma = nn.Linear(128, action_dim)

        torch.nn.init.xavier_uniform_(self.affine1.weight)
        torch.nn.init.xavier_uniform_(self.middle.weight)
        torch.nn.init.xavier_uniform_(self.mu.weight)
        torch.nn.init.xavier_uniform_(self.sigma.weight)

        self.saved_log_probs = []
        self.rewards = []

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env
        self.gamma = gamma
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.env.env.set_state(ant_utils.qpos, ant_utils.qvel)
        self.init_state = np.array(self.env.env.state_vector())
        self.env.seed(int(time.time())) # seed environment
        prng.seed(int(time.time())) # seed action space

    def init(self, init_policy):
        print("init to policy")
        self.load_state_dict(init_policy.state_dict())


    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.middle(x))
        mu = 2 * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x)) + 0.001      # avoid 0
        return mu, sigma


    def get_probs_and_var(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs, var = self.forward(state)
        return probs, var

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        mu, sigma = self.forward(state)
        m = Normal(mu, sigma)
        action = m.sample()
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
            policy_loss.append(-log_prob * reward.float())

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum() # cost function?
        policy_loss.backward()
        self.optimizer.step()
        self.rewards.clear()
        self.saved_log_probs.clear()

        return policy_loss


    def get_obs(self): # TODO
        return self.env.env.state_vector()

    def learn_policy(self, reward_fn, initial_state=[],episodes=1000, train_steps=1000):

        if len(initial_state) == 0:
            # initial_state = self.init_state
            initial_state = self.env.reset()
            initial_state = initial_state[:29]
        print("init: " + str(initial_state))

        qpos = initial_state[:15]
        qvel = initial_state[15:]

        running_reward = 0
        running_loss = 0
        for i_episode in range(episodes):
            # if i_episode % 2 == 0:
            #     self.env.env.set_state(qpos, qvel)
            self.env.reset()
            state = self.get_obs()
            ep_reward = 0
            for t in range(train_steps):  # Don't infinite loop while learning
                action = self.select_action(state)
                _, _, done, _ = self.env.step(action)
                state = self.get_obs()
                reward = reward_fn[tuple(ant_utils.discretize_state(state))]
                ep_reward += reward
                self.rewards.append(reward)
                if done:
                    self.env.reset()

            running_reward = running_reward * 0.99 + ep_reward * 0.01
            if (i_episode == 0):
                running_reward = ep_reward
            
            loss = self.update_policy()
            running_loss = running_loss * 0.99 + loss*.01

            # Log to console.
            if i_episode % 10 == 0:
                print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLoss: {:.2f}'.format(
                    i_episode, ep_reward, running_reward, running_loss))

    def execute_internal(self, env, T, state, render):
        p = np.zeros(shape=(tuple(ant_utils.num_states)))
        print("Simulation starting at = " + str(state))
        state = self.get_obs()
        for t in range(T):  
            action = self.select_action(state)
            _, reward, done, _ = self.env.step(action)
            state = self.get_obs()
            p[tuple(ant_utils.discretize_state(state))] += 1

            if render:
                env.render()
            if done:
                env.reset()
        env.close()
        return p

    def execute(self, T, initial_state=[], render=False, video_dir=''):
        p = np.zeros(shape=(tuple(ant_utils.num_states)))

        if len(initial_state) == 0:
            initial_state = self.env.reset()
            initial_state = initial_state[:29]

        # print("initial_state = " + str(initial_state))

        qpos = initial_state[:15]
        qvel = initial_state[15:]

        if video_dir != '' and render:
            wrapped_env = wrappers.Monitor(self.env, video_dir)
            wrapped_env.reset()
            wrapped_env.unwrapped.set_state(qpos, qvel)
            state = self.get_obs()
            p = self.execute_internal(wrapped_env, T, state, render)
        else:
            self.env.reset()
            self.env.env.set_state(qpos, qvel)
            state = self.get_obs()
            p = self.execute_internal(self.env, T, state, render)
        
        return p/float(T)

    def execute_random_internal(self, env, T, state, render):
        p = np.zeros(shape=(tuple(ant_utils.num_states)))
        for t in range(T):  
            r = random.random()
            action = -1
            if (r < 1/3.):
                action = 0
            elif r < 2/3.:
                action = 1
            # action = self.env.action_space.sample() # continuous actions
            _, reward, done, _ = env.step([action])
            state = self.get_obs()
            p[tuple(ant_utils.discretize_state(state))] += 1
            
            if render:
                env.render()
            if done:
                env.reset()
        env.close()
        return p

    def execute_random(self, T, initial_state=[], render=False, video_dir=''):
        p = np.zeros(shape=(tuple(ant_utils.num_states)))

        if len(initial_state) == 0:
            initial_state = self.env.reset() # get random starting location
            initial_state = initial_state[:29]

        qpos = initial_state[:15]
        qvel = initial_state[15:]

        if video_dir != '' and render:
            print("rendering env in execute_random()")
            wrapped_env = wrappers.Monitor(self.env, video_dir)
            wrapped_env.reset()
            wrapped_env.unwrapped.set_state(qpos, qvel)
            state = self.get_obs()
            p = self.execute_random_internal(wrapped_env, T, state, render)
        else:
            self.env.reset()
            self.env.env.set_state(qpos, qvel)
            state = self.get_obs()
            p = self.execute_random_internal(self.env, T, state, render)

        return p/float(T)

    def save(self, filename):
        self.env.close()
        torch.save(self, filename)