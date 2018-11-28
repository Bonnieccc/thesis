import numpy as np
import argparse

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')

parser.add_argument('--gamma', type=float, default=0.99, metavar='g',
                    help='learning rate')
parser.add_argument('--lr', type=float, default=1e-4, metavar='lr',
                    help='learning rate')
parser.add_argument('--eps', type=float, default=0.05, metavar='eps',
                    help='exploration rate')
parser.add_argument('--train_steps', type=int, default=2000, metavar='ts',
                    help='number of steps per episodes')
parser.add_argument('--episodes', type=int, default=5000, metavar='ep',
                    help='number of episodes per agent')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--model_count', type=int, default=50, metavar='mc',
                    help='number of models to train on entropy rewards')
parser.add_argument('--models_dir', type=str, default='models_cheetah/models_cheetah2018_11_26-11-58/', metavar='N',
                    help='directory from which to load model policies')
parser.add_argument('--env', type=str, default='HalfCheetah-v2', metavar='env',
                    help='the env to learn')
args = parser.parse_args()


# Env variables for MountainCarContinuous
nx = 10
nv = 4
mc_obs_dim = 2
mc_action_dim = 2

# Env variables for HalfCheetah
cheetah_num_bins = 10
cheetah_obs_dim = 5 #len(env.observation_space.high)
cheetah_action_dim = 6
cheetah_space_dim = cheetah_num_bins**cheetah_obs_dim


def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.asscalar(np.digitize(x=value, bins=bins))

#### Set up environment.

def get_state_bins():
    state_bins = []
    if args.env == "HalfCheetah-v2":
        for i in range(cheetah_obs_dim):
            state_bins.append(discretize_range(-2, 2, cheetah_num_bins))
    elif args.env == "MountainCarContinuous-v0":
        state_bins = [
            # Cart position.
            discretize_range(-1.2, 0.6, nx), 
            # Cart velocity.
            discretize_range(-0.07, 0.07, nv)
        ]
    return state_bins

def get_obs_dim():
    if args.env == "HalfCheetah-v2":
        return cheetah_obs_dim
    elif args.env == "MountainCarContinuous-v0":
        return mc_obs_dim

def get_action_dim():
    if args.env == "HalfCheetah-v2":
        return cheetah_action_dim
    elif args.env == "MountainCarContinuous-v0":
        return mc_action_dim

def get_space_dim():
    if args.env == "HalfCheetah-v2":
        return cheetah_space_dim
    elif args.env == "MountainCarContinuous-v0":
        return (nx, nv)


action_dim = get_action_dim()
obs_dim = get_obs_dim()
state_bins = get_state_bins()
space_dim = get_space_dim()

num_states = []
for i in range(obs_dim):
    num_states.append(len(state_bins[i]) + 1)

# Discretize the observation features and reduce them to a single list.
def discretize_state(observation):
    state = []
    for i, feature in enumerate(observation):
        if i >= obs_dim:
            break
        state.append(discretize_value(feature, state_bins[i]))
    return state



