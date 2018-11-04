import sys
import getopt
import numpy as np
import theano.tensor as T
from ilqr import iLQR
from ilqr.dynamics import FiniteDiffDynamics
from ilqr.cost import QRCost
from ilqr.cost import BatchAutoDiffCost
import gym
import math
import time
from gym.spaces import prng

class MountainCar():

    def __init__(self):
        self.env = gym.make("MountainCarContinuous-v0")
        self.env.seed(int(time.time())) # seed environment
        prng.seed(int(time.time())) # seed action space

        self.initial_state = self.env.reset()
        print("initial_state: " + str(self.initial_state))

        self.state_size = len(self.env.observation_space.sample())
        self.action_size = 1

        self.x_goal = np.array([0.50, 1])
        self.cost = 0

    def f(self, x, u, i):
        """Dynamics model function.

        Args:
            x: State vector [state_size].
            u: Control vector [action_size].
            i: Current time step.

        Returns:
            Next state vector [state_size].
        """
        self.env.env.state = x
        next_state, reward, done, info = self.env.step(u)
        self.cost += reward
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
            print("Final episode: lasted {} timesteps, reward = {}".format(steps, total_reward))
            break

    return total_reward, steps

def control_ilqr():
    cart = MountainCar()
    dynamics = FiniteDiffDynamics(cart.f, cart.state_size, cart.action_size)

    # The coefficients weigh how much your state error is worth to you vs
    # the size of your controls. You can favor a solution that uses smaller
    # controls by increasing R's coefficient.
    Q = 500 * np.eye(cart.state_size)
    R = 100 * np.eye(cart.action_size)

    Q_terminal = np.eye(cart.state_size)*1000
    cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=cart.x_goal)
    # cost = BatchAutoDiffCost(cart.cost_function, cart.state_size, cart.action_size)

    N = 1000  # Number of time-steps in trajectory.
    x0 = cart.initial_state  # Initial state.
    us_init = np.random.uniform(-1, 1, (N, 1)) # Random initial action path.

    ilqr = iLQR(dynamics, cost, N)
    xs, us = ilqr.fit(x0, us_init)
    cart.close()

    return us

def run_experiment(trials):
    threshold = 50.0
    rewards = []

    for trial in range(trials):
        print("--------------------")
        us = control_ilqr()
        render_env = gym.make("MountainCarContinuous-v0")
        reward, steps = simulate(render_env, us)
        render_env.close()
        rewards.append(reward)


    success = sum(float(r) >= threshold for r in rewards)
    failure = len(rewards) - success

    pct_success = success/float(len(rewards))
    pct_failure = failure/float(len(rewards))

    print("********************")
    print("% success: " + str(pct_success))
    print("% failure: " + str(pct_failure))


def main(argv):
    trials = 1
    opts_short = "n:"  
    opts_long = ["trials="]

    try:  
        args, values = getopt.getopt(argv, opts_short, opts_long)
    except getopt.error as err:  
        # output error, and return with an error code
        print (str(err))
        sys.exit(2)

    for arg, v in args:  
        if arg in ("-n", "--trials"):
            trials = int(v)
    
    run_experiment(trials)

if __name__ == "__main__":
    main(sys.argv[1:])
    






