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
    m = Normal(mu, sigma)
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
            prob, v = policy.get_probs_and_var(env.env.state_vector())
            probs += prob
            var += v
        probs /= len(policies)
        var /= len(policies)
        action = select_action(probs, var)
        
        state, reward, done, _ = env.step(action)
        p[tuple(ant_utils.discretize_state(state))] += 1
        if t == random_T:
            random_initial_state = env.env.state_vector()

        if render:
            env.render()
        if done:
            env.reset()

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
            env.reset()
            initial_state = env.env.state_vector()

        qpos = initial_state[:15]
        qvel = initial_state[15:]

        # NOTE: this records ONLY the final run. 
        if video_dir != '' and render and i == last_run:
            wrapped_env = wrappers.Monitor(env, video_dir)
            state = wrapped_env.reset()
            wrapped_env.unwrapped.set_state(qpos, qvel)
            
            p, random_initial_state = execute_policy_internal(wrapped_env, T, policies, state, True)
            average_p += p
            avg_entropy += scipy.stats.entropy(average_p.flatten())

        else:
            state = env.reset()
            env.env.set_state(qpos, qvel)
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
# Main loop of maximum entropy program. Iteratively collect 
# and learn T policies using policy gradients and a reward function 
# based on entropy.
def collect_entropy_policies(env, epochs, T, MODEL_DIR=''):

    reward_fn = np.zeros(shape=(tuple(ant_utils.num_states)))

    # set initial state to base state.
    seed = init_state(env)
    reward_fn[tuple(ant_utils.discretize_state(seed))] = 1
    print(seed)
    print(tuple(ant_utils.discretize_state(seed)))

    running_avg_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_ent = 0
    window_running_avg_p = np.zeros(shape=(tuple(ant_utils.num_states)))
    window_running_avg_ent = 0

    running_avg_p_baseline = np.zeros(shape=(tuple(ant_utils.num_states)))
    running_avg_ent_baseline = 0
    window_running_avg_p_baseline = np.zeros(shape=(tuple(ant_utils.num_states)))
    window_running_avg_ent_baseline = 0

    baseline_entropies = []
    baseline_ps = []
    entropies = []
    ps = []

    average_entropies = []
    average_ps = []

    running_avg_entropies = []
    running_avg_ps = []

    running_avg_entropies_baseline = []
    running_avg_ps_baseline = []

    window_running_avg_ents = []
    window_running_avg_ps = []
    window_running_avg_ents_baseline = []
    window_running_avg_ps_baseline = []

    policies = []
    initial_state = [] #init_state(env)

    for i in range(epochs):

        # Learn policy that maximizes current reward function.
        policy = Policy(env, args.gamma, args.lr, ant_utils.obs_dim, ant_utils.action_dim) 
        policy.learn_policy(reward_fn, initial_state, args.episodes, args.train_steps)
        policies.append(policy)

        # if args.save_models:
        #     policy.save(MODEL_DIR + 'model_' + str(i) + '.pt')

        # Get next distribution p by executing pi for T steps.
        # p_videos = 'cmp_videos/%sp_%d/'% (MODEL_DIR, i) 
        initial_state = []
        p = policy.execute(T, initial_state, render=args.render)
        
        a = 10 # average over this many rounds
        baseline_videos = 'cmp_videos/%sbaseline_%d/'% (MODEL_DIR, i) # note that MODEL_DIR has trailing slash
        entropy_videos = 'cmp_videos/%sentropy_%d/'% (MODEL_DIR, i)
        p_baseline = policy.execute_random(T, render=False, video_dir=baseline_videos) # args.episodes?
        round_entropy_baseline = scipy.stats.entropy(p_baseline.flatten())
        for av in range(a - 1):
            next_p_baseline = policy.execute_random(T)
            p_baseline += next_p_baseline
            # print(scipy.stats.entropy(next_p_baseline.flatten()))
            round_entropy_baseline += scipy.stats.entropy(next_p_baseline.flatten())
        p_baseline /= float(a)
        round_entropy_baseline /= float(a) # running average of the entropy
        
        # note: the entropy is p_baseline is not the same as the computed avg entropy
        # print("baseline compare:")
        # print(round_entropy_baseline) # running average
        # print(scipy.stats.entropy(p_baseline.flatten())) # entropy of final

        # reward_fn = grad_ent(p)

        round_entropy = scipy.stats.entropy(p.flatten())
        entropies.append(round_entropy)
        baseline_entropies.append(round_entropy_baseline)
        ps.append(p)
        baseline_ps.append(p_baseline)

        # Execute the cumulative average policy thus far.
        # Estimate distribution and entropy.
        average_p, round_avg_ent, initial_state = \
            execute_average_policy(env, policies, T, initial_state=initial_state, avg_runs=a, render=False, video_dir=entropy_videos)

        reward_fn = grad_ent(average_p)

        average_ps.append(average_p)
        average_entropies.append(round_avg_ent) 
        
        # Update running average.
        window = 5
        if (i < window): # add normally
            window_running_avg_ent = window_running_avg_ent * (i)/float(i+1) + round_avg_ent/float(i+1)
            window_running_avg_p = window_running_avg_ent * (i)/float(i+1) + average_p/float(i+1)
            window_running_avg_ent_baseline = window_running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
            window_running_avg_p_baseline = window_running_avg_p_baseline * (i)/float(i+1) + p_baseline/float(i+1)

        else:
            window_running_avg_ent = window_running_avg_ent + round_avg_ent/float(window) - average_entropies[i-5]/float(window)
            window_running_avg_p = window_running_avg_p + average_p/float(window) - average_ps[i-5]/float(window)
            
            window_running_avg_ent_baseline = window_running_avg_ent_baseline + round_entropy_baseline/float(window) - baseline_entropies[i-5]/float(window)
            window_running_avg_p_baseline = window_running_avg_p_baseline + p_baseline/float(window) - baseline_ps[i-5]/float(window)
        

        running_avg_ent = running_avg_ent * (i)/float(i+1) + round_avg_ent/float(i+1)
        running_avg_p = running_avg_p * (i)/float(i+1) + average_p/float(i+1)
        running_avg_entropies.append(running_avg_ent)
        running_avg_ps.append(running_avg_p)     

        # Update baseline running averages.
        running_avg_ent_baseline = running_avg_ent_baseline * (i)/float(i+1) + round_entropy_baseline/float(i+1)
        running_avg_p_baseline = running_avg_p_baseline * (i)/float(i+1) + p_baseline/float(i+1)
        running_avg_entropies_baseline.append(running_avg_ent_baseline)
        running_avg_ps_baseline.append(running_avg_p_baseline) 

        window_running_avg_ents.append(window_running_avg_ent)
        window_running_avg_ps.append(window_running_avg_p)
        window_running_avg_ents_baseline.append(window_running_avg_ent_baseline)
        window_running_avg_ps_baseline.append(window_running_avg_p_baseline)

        # print("p=")
        # print(p)
        # print("..........")
        # print("round_entropy = %f" % (round_entropy))

        print("---------------------")

        # print("average_p =") 
        # print(average_p)

        # print("..........")

        print("round_avg_ent[%d] = %f" % (i, round_avg_ent))
        print("running_avg_ent = %s" % running_avg_ent)
        print("window_running_avg_ent = %s" % window_running_avg_ent)

        print("..........")

        print("round_entropy_baseline[%d] = %f" % (i, round_entropy_baseline))
        print("running_avg_ent_baseline = %s" % running_avg_ent_baseline)
        print("window_running_avg_ent_baseline = %s" % window_running_avg_ent_baseline)

        print("----------------------")

        #plotting.heatmap(running_avg_p, average_p, i)

    # plotting.smear_lines(running_avg_ps, running_avg_ps_baseline)
    # plotting.running_average_entropy(running_avg_entropies, running_avg_entropies_baseline)
    # plotting.running_average_entropy_window(window_running_avg_ents, window_running_avg_ents_baseline, window)
    # plotting.difference_heatmap(running_avg_ps, running_avg_ps_baseline)

    # indexes = []
    # print('which indexes?')
    # for i in range(4):
    #     idx = input("index :")
    #     indexes.append(int(idx))
    # plotting.heatmap4(running_avg_ps, running_avg_ps_baseline, indexes)

    return policies


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

    policies = collect_entropy_policies(env, args.epochs, args.T)

    exploration_policy = average_policies(env, policies)
    if (args.collect_video):
        MODEL_DIR = ''
    # average_p = exploration_policy.execute(args.T, render=True, save_video_dir=MODEL_DIR+'videos/epoch_' + str(args.epochs) + '/')
    overall_avg_ent = scipy.stats.entropy(average_p.flatten())

    # average_p = curiosity.execute_average_policy(env, policies, args.T, render=True)

    print('*************')
    # print(np.reshape(average_p, ant_utils.space_dim))

    print("overall_avg_ent = %f" % overall_avg_ent)

    env.close()

    print("DONE")

if __name__ == "__main__":
    main()


