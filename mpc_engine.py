import os
import gym
import numpy as np
import time
import random
import math
import random_mpc as mpc
import matplotlib.pyplot as plt
from gym.spaces import prng

def run_pendulum_experiment(M=10, k=2000, max_steps=2000, trials=100, render=False):

    rewards = []

    for trial in range(trials):
        env = gym.make("Pendulum-v0")
        env._max_episode_steps = max_steps

        env.seed(int(time.time())) # seed environment
        prng.seed(int(time.time())) # seed action space

        reward, steps, success = mpc.control_random_mpc_pendulum(env, M, k, render)
        rewards.append(reward)
        env.close()

        # Make sure rendering window is shut.
        time.sleep(1.0)


    print("********************")
    print(rewards)

def run_mountaincar_cont_experiment(M=10, k=2000, max_steps=2000, trials=100, render=False):

    successes = []

    for trial in range(trials):
        env = gym.make("MountainCarContinuous-v0")
        env._max_episode_steps = max_steps

        env.seed(int(time.time())) # seed environment
        prng.seed(int(time.time())) # seed action space

        reward, steps, success = mpc.control_random_mpc_mountaincar_continuous(env, M, k, render)
        successes.append(success)
        env.close()

        # Make sure rendering window is shut.
        time.sleep(1.0)

    print (successes)

    pct_success = successes.count(True)/float(len(successes))
    pct_failure = successes.count(False)/float(len(successes))

    print("********************")
    print("% success: " + str(pct_success))
    print("% failure: " + str(pct_failure))

def run_mountaincar_discrete_experiment(M=20, k=2000, max_steps=2000, trials=100, render=False):

    successes = []

    for trial in range(trials):
        env = gym.make("MountainCar-v0")
        env._max_episode_steps = max_steps

        env.seed(int(time.time())) # seed environment
        prng.seed(int(time.time())) # seed action space

        reward, steps, success = mpc.control_random_mpc_mountaincar_discrete(env, M, k, render)
        successes.append(success)
        env.close()

        # Make sure rendering window is shut.
        time.sleep(1.0)

    print (successes)

    pct_success = successes.count(True)/float(len(successes))
    pct_failure = successes.count(False)/float(len(successes))

    print("********************")
    print("% success: " + str(pct_success))
    print("% failure: " + str(pct_failure))

def run_cartpole_experiment(M=20, k=100, max_steps=2000, trials=100, render=False):

    exp_steps = []
    step_threshold = 100

    for trial in range(trials):
        env = gym.make("CartPole-v1")
        env._max_episode_steps = max_steps

        env.seed(int(time.time())) # seed environment
        prng.seed(int(time.time())) # seed action space

        reward, steps, success = mpc.control_random_mpc_cartpole(env, M, k, render)
        exp_steps.append(steps)
        env.close()

        # Make sure rendering window is shut.
        time.sleep(1.0)


    successes = sum(s >= step_threshold for s in exp_steps)
    failures = len(exp_steps) - successes

    pct_success = successes/float(trials)
    pct_failure = failures/float(trials)

    print("********************")
    print (exp_steps)
    print("% success: " + str(pct_success))
    print("% failure: " + str(pct_failure))


def main():
    random.seed(time.time())
    np.random.seed(seed=int(time.time()))

    ## Discrete envs ##
    # run_cartpole_experiment(M=100, k=100)
    # run_mountaincar_discrete_experiment(render=True)

    ## Continuous envs ##
    run_mountaincar_cont_experiment()
    # run_pendulum_experiment()

if __name__ == "__main__":
    main()
