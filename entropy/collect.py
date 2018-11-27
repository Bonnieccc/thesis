import os
import sys
import time
from datetime import datetime
import logging
import argparse

import numpy as np
import gym
from gym.spaces import prng

from entropy_policy import EntropyPolicy
from curiosity import SimplePolicy
import utils
from utils import args

if args.env == "HalfCheetah-v2":
    Policy = EntropyPolicy
else:
    Policy = SimplePolicy


def average_policies(policies):
    state_dict = policies[0].state_dict()
    for i in range(1, len(policies)):
        for k, v in policies[i].state_dict().items():
            state_dict[k] += v

    for k, v in state_dict.items():
        state_dict[k] /= float(len(policies))

    return state_dict


def log_iteration(i, logger, p, reward_fn):
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
    grad_p[grad_p > 100] = 500
    return grad_p


# Iteratively collect and learn T policies using policy gradients and a reward
# function based on entropy.
def collect_entropy_policies(env, iterations, T, MODEL_DIR, logger):
    reward_fn = np.ones(shape=(tuple(utils.num_states)))
    policies = []
    for i in range(iterations):
        # Learn policy that maximizes current reward function.
        policy = Policy(env, args.gamma)
        policy.learn_policy(reward_fn, args.episodes, args.train_steps)

        # Get next distribution p by executing pi for T steps.
        p = policy.execute(T)

        log_iteration(i, logger, p, reward_fn)

        print("p=")
        print(np.reshape(p, utils.space_dim))
        print("reward_fn=")
        print(np.reshape(reward_fn, utils.space_dim))
        print("----------------------")

        reward_fn = grad_ent(p)
        policies.append(policy)

        # Save the policy
        policy.save(MODEL_DIR + 'model_' + str(i) + '.pt')

    return policies

def main():

    # Suppress scientific notation.
    np.set_printoptions(suppress=True, edgeitems=100)

    # Make environment.
    env = gym.make(args.env)
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    # Set up experiment variables.
    iterations = 50
    T = 10000

    # set up logging to file 
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

    MODEL_DIR = 'models_' + args.env + '/models_' + TIME + '/'
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    policies = collect_entropy_policies(env, iterations, T, MODEL_DIR, logger)

    # # obtain average policy.
    average_policy_state_dict = utils.average_policies(policies)
    exploration_policy.load_state_dict(average_policy_state_dict)
    average_p = exploration_policy.execute(T)
   
    log_iteration('average', logger, average_p, [])

    print('*************')
    print(np.reshape(average_p, utils.space_dim))

    env.close()

if __name__ == "__main__":
    main()


