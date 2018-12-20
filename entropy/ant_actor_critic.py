import numpy as np
import time
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.autograd import Variable

from gym.spaces import prng
from gym import wrappers
import ant_utils

pi = Variable(torch.FloatTensor([np.pi]))

def normal(x, mu, sigma_sq):
    a = (-1*(Variable(x)-mu).pow(2)/(2*sigma_sq)).exp()
    b = 1/(2*sigma_sq*pi.expand_as(sigma_sq)).sqrt()
    return a*b

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class AntActorCritic(nn.Module):
    def __init__(self, env, gamma, lr, obs_dim, action_dim):
        super(AntActorCritic, self).__init__()

        self.linear1 = nn.Linear(obs_dim, 200)
        self.lstm = nn.LSTMCell(200, 128)
        # Actor
        self.mu_linear = nn.Linear(128, action_dim)
        self.sigma_sq_linear = nn.Linear(128, action_dim)
        # Critic
        self.value_linear = nn.Linear(128, 1)

        # initialize weight
        self.apply(weights_init)
        self.mu_linear.weight.data = normalized_columns_initializer(
                    self.mu_linear.weight.data, 0.01)
        self.sigma_sq_linear.weight.data = normalized_columns_initializer(
                    self.sigma_sq_linear.weight.data, 0.01)
        self.mu_linear.bias.data.fill_(0)
        self.sigma_sq_linear.bias.data.fill_(0) 

        self.value_linear.weight.data = normalized_columns_initializer(
                        self.value_linear.weight.data, 1.0)
        self.value_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.train()

        self.saved_log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.eps = np.finfo(np.float32).eps.item()

        self.env = env
        self.gamma = gamma
        self.tau = 1.00
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.env.env.set_state(ant_utils.qpos, ant_utils.qvel)
        self.init_state = np.array(self.env.env.state_vector())
        self.env.seed(int(time.time())) # seed environment
        prng.seed(int(time.time())) # seed action space

    def init(self, init_policy):
        print("init to policy")
        self.load_state_dict(init_policy.state_dict())


    def forward(self, x, hx, cx):
        x = F.relu(self.linear1(a))
        x = x.view(-1, 200)
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        
        return self.value_linear(x), self.mu_linear(x), self.sigma_sq_linear(x), (hx, cx)

    def get_probs_and_var(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        _, mu, sigma, _ = self.forward(state)
        return mu, sigma

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        value, mu, sigma, _ = self.forward(state)

        sigma = F.softplus(sigma)
        eps = torch.randn(mu.size())

        # calculate the probability
        action = (mu + sigma.sqrt()*Variable(eps)).data
        prob = normal(action, mu, sigma)
        entropy = -0.5*((sigma_sq + 2*pi.expand_as(sigma)).log() + 1)

        self.saved_log_probs.append(m.log_prob(action))
        self.entropies.append(entropy)
        self.values.append(value)

        return action

    def update_policy(self):

        policy_loss = 0
        value_loss = 0
        R = Variable(R)
        gae = torch.zeros(1, 1)
        # calculate the rewards from the terminal state
        for i in reversed(range(len(self.rewards))):
            R = self.gamma * R + self.rewards[i]
            advantage = R - self.values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            # convert the data into xxx.data will stop the gradient
            delta_t = self.rewards[i] + self.gamma * \
                self.values[i + 1].data - self.values[i].data
            gae = gae * self.gamma * args.tau + delta_t

            # for Mujoco, entropy loss lower to 0.0001
            policy_loss = policy_loss - (self.saved_log_probs[i]*Variable(gae).expand_as(
                self.saved_log_probs[i])).sum() - (0.0001*self.entropies[i]).sum()

        self.optimizer.zero_grad()
        
        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)
        
        self.optimizer.step()

        self.rewards.clear()
        self.saved_log_probs.clear()
        self.entropies.clear()
        self.values.clear()

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