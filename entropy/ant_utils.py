import gym
import time
import numpy as np

env = gym.make('Ant-v2')

qpos = env.env.init_qpos
qvel = env.env.init_qvel

obs_dim = int(env.env.state_vector().shape[0])
action_dim = int(env.action_space.sample().shape[0])

features = [2, 3] # chosen at random
min_bin = -10
max_bin = 10
num_bins = 6

space_dim = (num_bins, num_bins) # should match len(features)

def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

#### Set up environment.

def get_state_bins():
    state_bins = []
    for i in range(len(features)):
        state_bins.append(discretize_range(min_bin, max_bin, num_bins))
    return state_bins


def get_num_states(state_bins):
    num_states = []
    for i in range(len(state_bins)):
        num_states.append(len(state_bins[i]) + 1)
    return num_states


state_bins = get_state_bins()
num_states = get_num_states(state_bins) # vector denoting the dimension of discretized state


# Discretize the observation features and reduce them to a single list.
def discretize_state(observation):
    state = []
    for i, idx in enumerate(features):
        state.append(discretize_value(observation[idx], state_bins[i]))
    return state









