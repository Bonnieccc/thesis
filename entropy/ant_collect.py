# Collect entropy-based reward policies.

# python collect.py --env="MountainCarContinuous-v0" --T=1000 --train_steps=400 --episodes=300 --epochs=50

import os
import time
from datetime import datetime

import numpy as np
import scipy.stats
from scipy.interpolate import interp2d
from scipy.interpolate import spline
from scipy.stats import norm

import gym
from gym.spaces import prng

from ant_entropy_policy import AntEntropyPolicy
import utils
import ant_utils
import curiosity
import plotting

import torch
from torch.distributions import Normal
import random

args = utils.get_args()
Policy = AntEntropyPolicy

def select_action(mu, sigma):
    m = Normal(mu.view(1, ).data, sigma.view(1, ).data)
    return m.sample().numpy()

def average_policies(env, policies):
    state_dict = policies[0].state_dict()
    for i in range(1, len(policies)):
        for k, v in policies[i].state_dict().items():
            state_dict[k] += v

    for k, v in state_dict.items():
        state_dict[k] /= float(len(policies))
     # obtain average policy.
    average_policy = Policy(env, args.gamma, args.lr, ant_utils.obs_dim, ant_utils.action_dim)
    average_policy.load_state_dict(state_dict)

    return average_policy

def execute_policy_internal(env, T, policies, state, render):
    random_T = np.floor(random.random()*T)
    p = np.zeros(shape=(tuple(ant_utils.num_states)))
    random_initial_state = []

    for t in range(T):
        # Compute average probability over action space for state.
        probs = torch.tensor(np.zeros(shape=(1,ant_utils.action_dim))).float()
        var = torch.tensor(np.zeros(shape=(1,ant_utils.action_dim))).float()
        for policy in policies:
            prob, v = policy.get_probs_and_var(state)
            probs += prob
            var += v
        probs /= len(policies)
        var /= len(policies)
        action = select_action(probs, var)
        
        state, reward, done, _ = env.step(action)
        p[tuple(ant_utils.discretize_state(state))] += 1
        if (t == random_T and not render):
            random_initial_state = env.env.state_vector()

        if render:
            env.render()
        if done:
            break # env.reset()

    p /= float(T)
    return p, random_initial_state

# run a simulation to see how the average policy behaves.
def execute_average_policy(env, policies, T, initial_state=[], avg_runs=1, render=False, video_dir=''):
    
    average_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    avg_entropy = 0
    random_initial_state = []

    last_run = avg_runs-1
    for i in range(avg_runs):
        if len(initial_state) == 0:
            initial_state = env.reset()

        # NOTE: this records ONLY the final run. 
        if video_dir != '' and render and i == last_run:
            wrapped_env = wrappers.Monitor(env, video_dir)
            wrapped_env.unwrapped.reset_state = initial_state
            state = wrapped_env.reset()
            # state = get_obs(state)
            
            p, random_initial_state = execute_policy_internal(wrapped_env, T, policies, state, True)
            average_p += p
            avg_entropy += scipy.stats.entropy(average_p.flatten())

        else:
            env.env.reset_state = initial_state
            state = env.reset()
            p, random_initial_state = execute_policy_internal(env, T, policies, state, False)
            average_p += p
            avg_entropy += scipy.stats.entropy(average_p.flatten())

    env.close()
    average_p /= float(avg_runs)

    avg_entropy /= float(avg_runs) # running average of the entropy 
    entropy_of_final = scipy.stats.entropy(average_p.flatten())

    # print("compare:")
    # print(avg_entropy) # running entropy
    # print(entropy_of_final) # entropy of the average distribution

    return average_p, avg_entropy, random_initial_state

def average_p_and_entropy(env, policies, T, avg_runs=1, render=False, video_dir=''):
    policy = average_policies(env, policies)
    average_p = policy.execute(T, render=render, video_dir=video_dir)
    average_ent = scipy.stats.entropy(average_p.flatten())
    for i in range(avg_runs-1):
        p = policy.execute(T)
        average_p += p
        average_ent += scipy.stats.entropy(p.flatten())
        # print(scipy.stats.entropy(p.flatten()))
    average_p /= float(avg_runs) 
    average_ent /= float(avg_runs)
    entropy_of_final = scipy.stats.entropy(average_p.flatten())

    # print("entropy compare: ")
    # print(average_ent) # running entropy
    # print(entropy_of_final) # entropy of final

    # return average_p, avg_entropy
    return average_p, average_ent


def grad_ent(pt):
    grad_p = -np.log(pt)
    grad_p[grad_p > 100] = 1000
    return grad_p

def init_state(env):
    env.env.set_state(ant_utils.qpos, ant_utils.qvel)
    return env.env.state_vector()

# Main loop of maximum entropy program. WORKING HERE
# Iteratively collect and learn T policies using policy gradients and a reward
# function based on entropy.
def collect_entropy_policies(env, epochs, T, MODEL_DIR=''):
    reward_fn = np.zeros(shape=(tuple(ant_utils.num_states)))
    
    # set initial state to base, motionless state.
    seed = init_state(env)
    reward_fn[tuple(ant_utils.discretize_state(seed))] = 1

    running_avg_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_ent = 0

    entropies = []
    average_ps = []

    running_avg_entropies = []
    running_avg_ps = []

    policies = []

    for i in range(epochs):

        env.env.state = init_state(env)

        # Learn policy that maximizes current reward function.
        policy = Policy(env, args.gamma, args.lr, ant_utils.obs_dim, ant_utils.action_dim)
        policy.learn_policy(reward_fn, args.episodes, args.train_steps)

        # Get next distribution p by executing pi for T steps.
        p = policy.execute(T, render=False)

        print("p=")
        print(np.reshape(p, ant_utils.space_dim))
        # print("reward_fn=")
        # print(np.reshape(reward_fn, utils.space_dim))

        # save the model
        policies.append(policy)
        #policy.save(MODEL_DIR + 'model_' + str(i) + '.pt')

        # model average policy.
        # average_policy = average_policies(env, policies)

        # average_p, round_avg_ent = average_p_and_entropy(env, policies, T)
        average_p, avg_entropy, random_initial_state = execute_average_policy(env, policies, T, render=False)
        
        # update rewards.
        reward_fn = grad_ent(p) # ORIGINAL/default
        if args.use_avg_reward_fn:
            reward_fn = grad_ent(average_p)
        
        print("average_p[0:%d]=" % i)
        print(np.reshape(average_p, ant_utils.space_dim))

        print("avg_entropy[0:%d] = %f" % (i, round_avg_ent))

        # alt_avg_p = curiosity.execute_average_policy(env, policies, T, render=False)
        # print("alt avg_entropy %d = %f" % (i, scipy.stats.entropy(alt_avg_p.flatten())))
        
        # Update running average.
        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_avg_ent/float(i+1)
        running_avg_p = running_avg_p * (i)/float(i+1) + average_p/float(i+1)

        # Save data from the round.
        entropies.append(round_avg_ent)
        average_ps.append(average_p)

        # Save the new running averages.
        running_avg_entropies.append(running_avg_ent)
        running_avg_ps.append(running_avg_p)       

        print("running_avg_ent = %s" % running_avg_ent)
        print("running_avg_p =") 
        print(running_avg_p)
        print("entropy: %s" % scipy.stats.entropy(running_avg_p.flatten()))
        print("----------------------")

    return policies, running_avg_entropies, entropies, running_avg_ps, average_ps

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up saving models.
    # TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
    # MODEL_DIR = 'models-' + args.env + '/models_' + TIME + '/'
    # if not os.path.exists(MODEL_DIR):
    #     os.makedirs(MODEL_DIR)

    # # save metadata from the run. 
    # with open(MODEL_DIR + "metadata", "w") as metadata:
    #     metadata.write("args: %s\n" % args)
    #     metadata.write("num_states: %s\n" % str(ant_utils.num_states))
    #     metadata.write("state_bins: %s\n" % ant_utils.state_bins)

    policies, running_avg_entropies, entropies, running_avg_ps, average_ps = collect_entropy_policies(env, args.epochs, args.T)
    #plotting.generate_figures(args.env, MODEL_DIR, running_avg_entropies, entropies, running_avg_ps, average_ps)

    exploration_policy = average_policies(env, policies)
    if (args.collect_video):
        MODEL_DIR = ''
    # average_p = exploration_policy.execute(args.T, render=True, save_video_dir=MODEL_DIR+'videos/epoch_' + str(args.epochs) + '/')
    overall_avg_ent = scipy.stats.entropy(average_p.flatten())

    # average_p = curiosity.execute_average_policy(env, policies, args.T, render=True)

    print('*************')
    print(np.reshape(average_p, ant_utils.space_dim))

    print("overall_avg_ent = %f" % overall_avg_ent)

    env.close()

    print("DONE")

if __name__ == "__main__":
    main()


