import sys
import getopt
import numpy as np
import theano.tensor as T
from ilqr import iLQR
from ilqr.dynamics import FiniteDiffDynamics
from ilqr.cost import QRCost
import gym
import math
import time
from gym.spaces import prng

class Pendulum():

    def __init__(self):
        self.env = gym.make("Pendulum-v0")
        self.env.seed(int(time.time())) # seed environment
        prng.seed(int(time.time())) # seed action space

        self.initial_state = self.env.reset()

        self.state_size = len(self.env.observation_space.sample())
        self.action_size = 1

        self.x_goal = np.array([np.sin(0), np.cos(0), 0])

    def f(self, x, u, i):
        """Dynamics model function.

        Args:
            x: State vector [state_size].
            u: Control vector [action_size].
            i: Current time step.

        Returns:
            Next state vector [state_size].
        """
        theta = np.arctan2(x[0], x[1])
        theta_dot = x[2]
        self.env.env.state = np.hstack([theta, theta_dot])
        next_state, reward, done, info = self.env.step(u)
        return next_state

    def close(self):
        self.env.close()

def simulate(env, trajectory, render=False):
    obs = env.reset()
    total_reward = 0
    steps = 0
    for u in trajectory:
        steps += 1
        if render:
            env.render()
            time.sleep(0.05) # slows down process to make it more visible
        obs, reward, done, info = env.step(u)
        total_reward += reward
        if done:
            print("Episode: lasted {} timesteps, reward = {}".format(steps, total_reward))
            break

    return total_reward, steps

# N = Number of time-steps in trajectory.
def control_ilqr(N=1000):
    pend = Pendulum()
    dynamics = FiniteDiffDynamics(pend.f, pend.state_size, pend.action_size)

    # The coefficients weigh how much your state error is worth to you vs
    # the size of your controls. You can favor a solution that uses smaller
    # controls by increasing R's coefficient.
    Q = 100 * np.eye(pend.state_size)
    R = 50 * np.eye(pend.action_size)

    Q_terminal = np.diag((1000, 1000, 1000))
    cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=pend.x_goal)

    x0 = pend.initial_state  # Initial state.
    us_init = np.random.uniform(-2, 2, (N, 1)) # Random initial action path.

    ilqr = iLQR(dynamics, cost, N)
    xs, us = ilqr.fit(x0, us_init)
    pend.close()

    return us

def run_experiment(trials, steps, render):
    rewards = []

    for trial in range(trials):
        print("--------------------")
        print("trial " + str(trial))
        us = control_ilqr(steps)
        render_env = gym.make("Pendulum-v0")
        reward, steps = simulate(render_env, us, render)
        render_env.close()
        rewards.append(reward)

    print(rewards)


def main(argv):
    trials = 1
    steps = 1
    render = False

    opts_short = "t:n:r"
    opts_long = ["trials=", "steps=", "render"]

    try:  
        args, values = getopt.getopt(argv, opts_short, opts_long)
    except getopt.error as err:  
        # output error, and return with an error code
        print (str(err))
        sys.exit(2)

    for arg, v in args:  
        if arg in ("-t", "--trials"):
            trials = int(v)
        if arg in ("-n", "--steps"):
            steps = int(v)
        if arg in ("-r", "--render"):
            render = True
    
    print("Running {} experiments with {} steps each.".format(trials, steps))
    run_experiment(trials, steps, render)

if __name__ == "__main__":
    main(sys.argv[1:])





