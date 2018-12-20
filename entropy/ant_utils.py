# self.sim.data.qpos are the positions, with the first 7 element the 
# 3D position (x,y,z) and orientation (quaternion x,y,z,w) of the torso, 
# and the remaining 8 positions are the joint angles.

# The [2:], operation removes the first 2 elements from the position, 
# which is the X and Y position of the agent's torso.

# self.sim.data.qvel are the velocities, with the first 6 elements 
# the 3D velocity (x,y,z) and 3D angular velocity (x,y,z) and the 
# remaining 8 are the joint velocities.

# 0 - x position
# 1 - y position
# 2 - z position
# 3 - x torso orientation
# 4 - y torso orientation
# 5 - z torso orientation
# 6 - w torso orientation
# 7-14 - joint angles

# 15-21 - 3d velocity/angular velocity
# 23-29 - joint velocities


import gym
import time
import numpy as np

env = gym.make('Ant-v2')

qpos = env.env.init_qpos
qvel = env.env.init_qvel

obs_dim = int(env.env.state_vector().shape[0])
action_dim = int(env.action_space.sample().shape[0])

features = [2,7,8,9,10]
min_bin = -5
max_bin = 5
num_bins = 7

space_dim = (num_bins, num_bins) # should match len(features)

def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

#### Set up environment.

def get_state_bins():
    state_bins = [
        # height
        discretize_range(0, 5, num_bins),
        # other fields
        discretize_range(min_bin, max_bin, num_bins),
        discretize_range(min_bin, max_bin, num_bins),
        discretize_range(min_bin, max_bin, num_bins),
        discretize_range(min_bin, max_bin, num_bins)
    ]
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









