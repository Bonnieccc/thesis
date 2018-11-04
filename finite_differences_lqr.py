import os
import gym
import numpy as np
import time
import spectral_filtering as sf
from sklearn.metrics import mean_squared_error
import control
import scipy.linalg
import copy
from gym.spaces import prng

def lqr(A,B,Q,R):
    # Solves for the optimal infinite-horizon LQR gain matrix given linear system (A,B) 
    # and cost function parameterized by (Q,R)
    M = []
    try:
        M = scipy.linalg.solve_continuous_are(A,B,Q,R)
    except np.linalg.LinAlgError as e:
        return M,True # catch errors

    # K=(B'MB + R)^(-1)*(B'MA)
    return np.dot(scipy.linalg.inv(np.dot(np.dot(B.T,M),B)+R),(np.dot(np.dot(B.T,M),A))), False


def simulate_env(copy_env, state, action):
    set_state = state
    if (len(state) != len(copy_env.env.state)):
        set_state = [np.arccos(min(max(state[0], -1), 1)), state[2]]

    copy_env.env.state = set_state
    next_state, reward, done, info = copy_env.step(action)

    # calculate the change in state
    xdot = ((next_state - state) / action).squeeze()
    return xdot


# Finite differences for continuous action spaces.
def finite_differences(env, state, action, discrete=False):

    eps = 0.01
    A = np.zeros((len(state), len(state)))
    for ii in range(len(state)):
        copy_env_inc = copy.deepcopy(env)
        x = state.copy()
        x[ii] += eps
        x_inc = simulate_env(copy_env_inc, state=x, action=action)
        copy_env_inc.close()

        copy_env_dec = copy.deepcopy(env)
        x = state.copy()
        x[ii] -= eps
        x_dec = simulate_env(copy_env_dec, state=x, action=action)
        copy_env_dec.close()

        A[:,ii] = (x_inc - x_dec) / (2 * eps)

    B = np.zeros((len(state), len(action)))
    for ii in range(len(action)):
        copy_env_inc = copy.deepcopy(env)
        u = action.copy()
        u[ii] += eps
        x_inc = simulate_env(copy_env_inc, state=state, action=u)
        copy_env_inc.close()

        copy_env_dec = copy.deepcopy(env)
        u = action.copy()
        u[ii] -= eps
        x_dec = simulate_env(copy_env_dec, state=state, action=u)
        copy_env_dec.close()

        B[:,ii] += (x_inc - x_dec) / (2 * eps)

    return A, B

def control_lqr_finite_differences(env, steps=500, render=False): 

    # target_state = [np.cos(np.pi/4), np.sin(np.pi/4), 0]

    obs = env.reset()
    action = env.action_space.sample()

    A, B = finite_differences(env, obs, action)
    
    Q = np.eye(len(obs))*1000
    R = 1
    K,_ = lqr(A, B, Q, R)

    print("K = " + str(K))
    print("Initial action: " + str(action))

    steps_lasted = 0
    total_reward = 0
    action_sequence = []
    for i in range(steps):
        steps_lasted += 1

        if (i % 50 == 0):
            A, B = finite_differences(env, obs, action)
            Q = np.eye(len(obs))*1000
            R = 1
            new_K,err = lqr(A, B, Q, R)
            if (not err):
                K = new_K

        action = np.dot(-K, obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        action_sequence.append(action)

        if done:
            print("Final episode: lasted {} timesteps, data: {}".format(i+1, obs))
            break

    if (render):
        obs = env.reset()
        for action in action_sequence:
            env.render()
            time.sleep(0.05) # slows down process to make it more visible
            obs, reward, done, info = env.step(action)

    return total_reward, steps_lasted, steps_lasted<steps

def run_experiment(env_str, trials=10, max_steps=2000, render=False):

    successes = []

    for trial in range(trials):
        print("--------------------------")
        env = gym.make(env_str)
        env._max_episode_steps = max_steps

        env.seed(int(time.time())) # seed environment
        prng.seed(int(time.time())) # seed action space

        reward, steps, success = control_lqr_finite_differences(env, max_steps, render)
        print("Reward = {}".format(reward))
        successes.append(success)
        env.close()

        # Make sure rendering window is shut.
        time.sleep(1.0)

    pct_success = successes.count(True)/float(len(successes))
    pct_failure = successes.count(False)/float(len(successes))

    print("********************")
    print (successes)
    print("% success: " + str(pct_success))
    print("% failure: " + str(pct_failure))

def main():
    ## Control with finite differences lqr ##
    run_experiment("MountainCarContinuous-v0")
    # run_experiment("Pendulum-v0", render=True)
    # run_experiment("CartPole-v1")

if __name__ == "__main__":
    main()
