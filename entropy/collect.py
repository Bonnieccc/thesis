# Collect entropy-based reward policies.

# Changed from using all-1 reward to init to one-hot at: 2018_11_30-10-00
# TODO: weights initialization?

# python collect.py --env="MountainCarContinuous-v0" --T=1000 --train_steps=400 --episodes=300 --epochs=50

import os
import time
from datetime import datetime
import logging

import numpy as np
import scipy.stats
import gym
from gym.spaces import prng

from cheetah_entropy_policy import CheetahEntropyPolicy
from cart_entropy_policy import CartEntropyPolicy
import utils
import curiosity

import torch
from torch.distributions import Normal
import random

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

args = utils.get_args()

Policy = CartEntropyPolicy
if args.env == "HalfCheetah-v2":
    Policy = CheetahEntropyPolicy

def select_action(probs, var):
    m = Normal(probs.detach(), var) 
    action = m.sample().numpy()[0]
    return action

def execute_average_policy(env, policies, T, render=False):
    # run a simulation to see how the average policy behaves.
    p = np.zeros(shape=(tuple(utils.num_states)))
    state = env.reset()
    for i in range(T):
        # Compute average probability over action space for state.
        probs = torch.tensor(np.zeros(shape=(1,utils.action_dim))).float()
        var = torch.tensor(np.zeros(shape=(1,utils.action_dim))).float()
        for policy in policies:
            prob = policy.get_probs(state)
            probs += prob
            # var += v
            
        probs /= len(policies)
        # var /= len(policies)
        action = select_action(probs, var)
        
        state, reward, done, _ = env.step(action)
        p[tuple(utils.discretize_state(state))] += 1

        if render and i % 10 == 0:
            env.render()
            time.sleep(.05)
        if done:
            env.reset()

    env.close()
    return p / float(T)

def average_policies(env, policies):
    state_dict = policies[0].state_dict()
    for i in range(1, len(policies)):
        for k, v in policies[i].state_dict().items():
            state_dict[k] += v

    for k, v in state_dict.items():
        state_dict[k] /= float(len(policies))
     # obtain average policy.
    average_policy = Policy(env, args.gamma, utils.obs_dim, utils.action_dim)
    average_policy.load_state_dict(state_dict)

    return average_policy


def log_iteration(i, logger, p, reward_fn):

    if isinstance(utils.space_dim, int):
        np.set_printoptions(suppress=True, threshold=utils.space_dim)

    if i == 'average':
        logger.debug("*************************")

    logger.debug('Iteration: ' + str(i))
    logger.debug('p' + str(i) + ':')
    logger.debug(np.reshape(p, utils.space_dim))

    if i != 'average':
        logger.debug('reward_fn' + str(i) + ':')
        logger.debug(np.reshape(reward_fn, utils.space_dim))

    np.set_printoptions(suppress=True, threshold=100, edgeitems=100)


def grad_ent(pt):
    grad_p = -np.log(pt)
    grad_p[grad_p > 100] = 1000
    return grad_p

def init_state(env_str):
    if args.env == "Pendulum-v0":
        return [np.pi, 0] # WORKING HERE
    elif args.env == "MountainCarContinuous-v0":
        return [-0.50, 0]

# Main loop of maximum entropy program. WORKING HERE
# Iteratively collect and learn T policies using policy gradients and a reward
# function based on entropy.
def collect_entropy_policies(env, epochs, T, MODEL_DIR, logger):
    reward_fn = -1*np.ones(shape=(tuple(utils.num_states)))

    # set initial state to base, motionless state.
    seed = []
    if args.env == "Pendulum-v0":
        env.env.state = [np.pi, 0] # WORKING HERE
        seed = env.env._get_obs()
    elif args.env == "MountainCarContinuous-v0":
        env.env.state = [-0.50, 0]
        seed = env.env.state

    print(tuple(utils.discretize_state(seed)))
    reward_fn[tuple(utils.discretize_state(seed))] = 1

    running_avg_p = np.zeros(shape=(tuple(utils.num_states)))
    running_avg_ent = 0

    entropies = []
    average_ps = []

    running_avg_entropies = []
    running_avg_ps = []

    policies = []

    for i in range(epochs):

        env.env.state = init_state(args.env)

        # Learn policy that maximizes current reward function.
        policy = Policy(env, args.gamma, utils.obs_dim, utils.action_dim)
        policy.learn_policy(reward_fn, args.episodes, args.train_steps)

        # Get next distribution p by executing pi for T steps.
        p = policy.execute(T, render=False)

        log_iteration(i, logger, p, reward_fn)

        print("p=")
        print(np.reshape(p, utils.space_dim))
        # print("reward_fn=")
        # print(np.reshape(reward_fn, utils.space_dim))

        # save the model
        policies.append(policy)
        policy.save(MODEL_DIR + 'model_' + str(i) + '.pt')

        # model average policy.
        # average_policy = average_policies(env, policies)

        save_video_dir = ''
        # if (i % 20 == 0): # This causes the pyplot library to fail for some reason
        #     save_video_dir = MODEL_DIR+'videos/epoch' +str(i) +'/' 

        average_p, round_avg_ent = curiosity.average_p_and_entropy(env, policies, T, save_video_dir=save_video_dir)
        
        # update rewards.
        reward_fn = grad_ent(p) # ORIGINAL/default
        if utils.args.use_avg_reward_fn:
            reward_fn = grad_ent(average_p)
        
        print("average_p[0:%d]=" % i)
        print(np.reshape(average_p, utils.space_dim))

        print("avg_entropy[0:%d] = %f" % (i, round_avg_ent))

        # alt_avg_p = execute_average_policy(env, policies, T, render=False)
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

        if i % 10 == 0 and args.render:
            avg_policy = average_policies(env, policies)
            avg_policy.execute(T, render=True)

    # # compute moving average with window N
    # N = len(policies)
    # running_avg_entropies = []
    # for i in range(1, len(policies) + 1):
    #     cumsum = np.cumsum(np.insert(entropies[0:i], 0, 0)) 
    #     cum = (cumsum[i:] - cumsum[:-i]) / float(i)
    #     print(cum)
    #     running_avg_entropies.append(cum)


    # # save file
    FIG_DIR = 'figs/' + args.env + '/'
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)
    model_time = MODEL_DIR.split('/')[1]
    fname = curiosity.get_next_file(FIG_DIR, model_time, "_running_avg_entropy", ".png")
    
    # plt.figure(1)
    # plt.plot(np.arange(epochs), running_avg_entropies)
    # plt.xlabel("t")
    # plt.ylabel("Running average entropy of cumulative policy")
    # plt.savefig(fname)
    # plt.show()

    # fname = curiosity.get_next_file(FIG_DIR, model_time, "_entropy", ".png")    
    # plt.figure(2)
    # plt.plot(np.arange(epochs), entropies)
    # plt.xlabel("t")
    # plt.ylabel("Entropy of policy t")
    # plt.savefig(fname)
    # plt.show()

    # want to plot the running_avg_p x_distribution over time
    plt.figure(3)
    ax_x = plt.subplot(211)
    ax_v = plt.subplot(212)

    ax_x.set_xlabel('t')
    ax_v.set_xlabel('t')
    ax_x.set_ylabel('Policy distribution over x')
    ax_v.set_ylabel('Policy distribution over v')

    fig_test = plt.figure(4)
    test = plt.subplot(111)

    for t in range(len(running_avg_ps)):
        running_avg_p = running_avg_ps[t]
        x_distribution = np.sum(running_avg_p, axis=1)
        v_distribution = np.sum(running_avg_p, axis=0)

        alphas_x = x_distribution
        colors_x = np.zeros((x_distribution.shape[0],4))
        colors_x[:, 3] = alphas_x

        alphas_v = v_distribution
        colors_v = np.zeros((v_distribution.shape[0],4))
        colors_v[:, 3] = alphas_v

        ax_x.scatter(t*np.ones(shape=x_distribution.shape), x_distribution, color=colors_x)
        ax_v.scatter(t*np.ones(shape=v_distribution.shape), v_distribution, color=colors_v)

        states = np.arange(x_distribution.shape[0])
        alphas = x_distribution.flatten()

        # expand data
        ls = np.linspace(0, x_distribution.shape[0], 100)
        estimate = np.interp(ls, states, alphas)

        colors = np.zeros((len(estimate),4))
        colors[:, 3] = estimate

        test.scatter(t*np.ones(shape=(len(estimate),1)), ls, color=colors)

    fname = curiosity.get_next_file(FIG_DIR, model_time, "_running_avg_xv_distrs_smear", ".png")
    fig_test.savefig(fname)
    fig_test.show()

    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up logging to file 
    TIME = datetime.now().strftime('%Y_%m_%d-%H-%M')
    LOG_DIR = 'logs-' + args.env + '/'
    if not os.path.exists(LOG_DIR):
        os.mkdir(LOG_DIR)

    FILE_NAME = 'test' + TIME
    logging.basicConfig(level=logging.DEBUG,
                        format='%(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=LOG_DIR + FILE_NAME + '.log',
                        filemode='w')
    logger = logging.getLogger(args.env + '-curiosity.pt')

    MODEL_DIR = 'models-' + args.env + '/models_' + TIME + '/'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # save metadata from the run. 
    with open(MODEL_DIR + "metadata", "w") as metadata:
        metadata.write("args: %s\n" % args)
        metadata.write("num_states: %s\n" % str(utils.num_states))
        metadata.write("state_bins: %s\n" % utils.state_bins)

    policies = collect_entropy_policies(env, args.epochs, args.T, MODEL_DIR, logger)

    exploration_policy = average_policies(env, policies)
    if (args.collect_video):
        MODEL_DIR = ''
    # average_p = exploration_policy.execute(args.T, render=True, save_video_dir=MODEL_DIR+'videos/epoch_' + str(args.epochs) + '/')
    overall_avg_ent = scipy.stats.entropy(average_p.flatten())

    # average_p = execute_average_policy(env, policies, args.T, render=True)

    log_iteration('average', logger, average_p, [])
    print('*************')
    print(np.reshape(average_p, utils.space_dim))

    print("overall_avg_ent = %f" % overall_avg_ent)

    env.close()

    print("DONE")

if __name__ == "__main__":
    main()


