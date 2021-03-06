# Original code author: Yuriy Guts
# Github link: https://github.com/YuriyGuts/cartpole-q-learning/blob/master/cartpole.py

import os
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import spectral_filtering as sf
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
import control
import scipy.linalg
import copy


class CartPoleQLearningAgent:
    def __init__(self,
                 learning_rate=0.2,
                 discount_factor=1.0,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99):

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.state = None
        self.action = None

        # Discretize the continuous state space for each of the 4 features.
        self._state_bins = [
            # Cart position.
            self._discretize_range(-2.4, 2.4, 2), 
            # Cart velocity.
            self._discretize_range(-2.0, 2.0, 2),
            # Pole angle.
            self._discretize_range(-0.5, 0.5, 6),
            # Rotation rate of pole.
            self._discretize_range(-2.0, 2.0, 3)
        ]

        # Create a clean Q-Table.
        self._num_actions = 2
        self._max_bins = max(len(bin) for bin in self._state_bins)
        num_states = [(len(self._state_bins[0]) + 1),
                      (len(self._state_bins[1]) + 1),
                      (len(self._state_bins[2]) + 1),
                      (len(self._state_bins[3]) + 1)]

        # want to have a q table that discretizes properly
        self.q = np.zeros(shape=(num_states[0], 
                                 num_states[1], 
                                 num_states[2], 
                                 num_states[3],
                                 self._num_actions))

    @staticmethod
    def _discretize_range(lower_bound, upper_bound, num_bins):
        return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

    @staticmethod
    def _discretize_value(value, bins):
        return np.asscalar(np.digitize(x=value, bins=bins))

    def _build_state(self, observation):
        # Discretize the observation features and reduce them to a single list.
        state = []
        for i, feature in enumerate(observation):
            state.append(self._discretize_value(feature, self._state_bins[i]))

        return state

    def begin_episode(self, observation):
        # Reduce exploration over time.
        self.exploration_rate *= self.exploration_decay_rate

        # Get the action for the initial state.
        self.state = self._build_state(observation)
        return np.argmax(self.q[tuple(self.state)])

    def act(self, observation, reward):
        next_state = self._build_state(observation)

        # Exploration/exploitation: choose a random action or select the best one.
        enable_exploration = (1 - self.exploration_rate) <= np.random.uniform(0, 1)
        if enable_exploration:
            next_action = np.random.randint(0, self._num_actions)
        else:
            next_action = np.argmax(self.q[tuple(next_state)])

        # Learn: update Q-Table based on current reward and future action.
        self.q[tuple(self.state) + (self.action,)] += \
            self.learning_rate * \
            (reward + self.discount_factor * \
            max(self.q[tuple(next_state) + (0,)], self.q[tuple(next_state) + (1,)]) \
            - self.q[tuple(self.state) + (self.action,)])

        self.state = next_state
        self.action = next_action
        return next_action

    # act according to the policy described in the Q table.
    def act_policy(self, observation):
        next_state = self._build_state(observation)
        next_action = np.argmax(self.q[tuple(next_state)])

        self.state = next_state
        self.action = next_action
        return next_action


class EpisodeHistory:
    def __init__(self,
                 capacity,
                 plot_episode_count=200,
                 max_timesteps_per_episode=200,
                 goal_avg_episode_length=195,
                 goal_consecutive_episodes=100):

        self.lengths = np.zeros(capacity, dtype=int)
        self.plot_episode_count = plot_episode_count
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.goal_avg_episode_length = goal_avg_episode_length
        self.goal_consecutive_episodes = goal_consecutive_episodes

        self.point_plot = None
        self.mean_plot = None
        self.fig = None
        self.ax = None

        # Record history of (state, action) tuples

    def __getitem__(self, episode_index):
        return self.lengths[episode_index]

    def __setitem__(self, episode_index, episode_length):
        self.lengths[episode_index] = episode_length

    def create_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(14, 7), facecolor='w', edgecolor='k')
        self.fig.canvas.set_window_title("Episode Length History")

        self.ax.set_xlim(0, self.plot_episode_count + 5)
        self.ax.set_ylim(0, self.max_timesteps_per_episode + 5)
        self.ax.yaxis.grid(True)

        self.ax.set_title("Episode Length History")
        self.ax.set_xlabel("Episode #")
        self.ax.set_ylabel("Length, timesteps")

        self.point_plot, = plt.plot([], [], linewidth=2.0, c="#1d619b")
        self.mean_plot, = plt.plot([], [], linewidth=3.0, c="#df3930")

    def update_plot(self, episode_index):
        plot_right_edge = episode_index
        plot_left_edge = max(0, plot_right_edge - self.plot_episode_count)

        # Update point plot.
        x = range(plot_left_edge, plot_right_edge)
        y = self.lengths[plot_left_edge:plot_right_edge]
        self.point_plot.set_xdata(x)
        self.point_plot.set_ydata(y)
        self.ax.set_xlim(plot_left_edge, plot_left_edge + self.plot_episode_count)

        # Update rolling mean plot.
        mean_kernel_size = 101
        rolling_mean_data = np.concatenate((np.zeros(mean_kernel_size), self.lengths[plot_left_edge:episode_index]))
        rolling_mean_data = pd.Series(rolling_mean_data)

        rolling_means = rolling_mean_data.rolling(mean_kernel_size,min_periods=0).mean()[mean_kernel_size:]

        self.mean_plot.set_xdata(range(plot_left_edge, plot_left_edge + len(rolling_means)))
        self.mean_plot.set_ydata(rolling_means)

        # Repaint the surface.
        plt.draw()
        plt.pause(0.0001)

    def is_goal_reached(self, episode_index):
        avg = np.average(self.lengths[episode_index - self.goal_consecutive_episodes + 1:episode_index + 1])
        return avg >= self.goal_avg_episode_length

    def close(self):
        plt.close(self.fig)


def log_timestep(index, action, reward, observation):
    format_string = "   ".join([
        "Timestep: {0:3d}",
        "Action: {1:2d}",
        "Reward: {2:5.1f}",
        "Cart Position: {3:6.3f}",
        "Cart Velocity: {4:6.3f}",
        "Angle: {5:6.3f}",
        "Tip Velocity: {6:6.3f}"
    ])
    print(format_string.format(index, action, reward, *observation))

def run_agent(env, verbose=False):
    max_episodes_to_run = 5000
    max_timesteps_per_episode = 500

    goal_avg_episode_length = 495
    goal_consecutive_episodes = 300

    plot_episode_count = 200
    plot_redraw_frequency = 10

    agent = CartPoleQLearningAgent(
        learning_rate=0.05,
        discount_factor=0.95,
        exploration_rate=0.15,
        exploration_decay_rate=0.99
    )

    episode_history = EpisodeHistory(
        capacity=max_episodes_to_run,
        plot_episode_count=plot_episode_count,
        max_timesteps_per_episode=max_timesteps_per_episode,
        goal_avg_episode_length=goal_avg_episode_length,
        goal_consecutive_episodes=goal_consecutive_episodes
    )
    episode_history.create_plot()

    for episode_index in range(max_episodes_to_run):
        observation = env.reset()
        action = agent.begin_episode(observation)

        for timestep_index in range(max_timesteps_per_episode):
            # Perform the action and observe the new state.
            observation, reward, done, info = env.step(action)

            # Update the display and log the current state.
            if verbose:
                env.render()
                log_timestep(timestep_index, action, reward, observation)

            # If the episode has ended prematurely, penalize the agent.
            if done and timestep_index < max_timesteps_per_episode - 1:
                reward = -max_episodes_to_run

            # Get the next action from the learner, given our new state.
            action = agent.act(observation, reward)

            # Record this episode to the history and check if the goal has been reached.
            if done or timestep_index == max_timesteps_per_episode - 1:
                print("Episode {} finished after {} timesteps.".format(episode_index + 1, timestep_index + 1))

                episode_history[episode_index] = timestep_index + 1
                if verbose or episode_index % plot_redraw_frequency == 0:
                    episode_history.update_plot(episode_index)

                if episode_history.is_goal_reached(episode_index):
                    print()
                    print("Goal reached after {} episodes!".format(episode_index + 1))

                    return agent, episode_history

                break

    print("Goal not reached after {} episodes.".format(max_episodes_to_run))
    return episode_history

def save_history(history, experiment_dir):
    # Save the episode lengths to CSV.
    filename = os.path.join(experiment_dir, "episode_history.csv")
    dataframe = pd.DataFrame(history.lengths, columns=["length"])
    dataframe.to_csv(filename, header=True, index_label="episode")


def collect_episode(env, agent, T=500, discretize=False):

    X = []
    Y = []
    actions = []

    state = env.reset()
    action = agent.begin_episode(state)
    t = 0
    while t < T:
        actions.append(action)
        # Get the next action from the learner, given our new state.
        next_state, reward, done, info = env.step(action)
        action = agent.act_policy(next_state)

        # Construct trial matrices X and Y
        if discretize:
            X.append(agent._build_state(state))
            Y.append(agent._build_state(next_state))
        else:
            X.append(state)
            Y.append(next_state)

        if done:
            state = env.reset()
            action = agent.begin_episode(next_state)

        state = next_state
        t = t + 1

    print("Trial episode: lasted {} timesteps".format(t))

    X = np.array(X).T # list to numpy matrix
    Y = np.array(Y).T # list to numpy matrix
    return X,Y, actions


def run_spectral_filtering(env, agent, k=20, T=500, num_trials=5):

    avg_losses = np.zeros(shape=(T))

    for trial in range(num_trials):

        X,Y,_ = collect_episode(env, agent, T)

        # compute eigenpairs
        vals, vecs = sf.eigen_pairs(T, k)

        # run wave filtering on episode data.
        avg_losses += sf.wave_filter(X, Y, k, vals, vecs,verbose=True)

    return avg_losses/num_trials

def get_control(K, state):
    val = np.dot(-K, state)
    action = 0 if (val < 0) else 1
    return action

def lqr(A,B,Q,R):
    # Solves for the optimal infinite-horizon LQR gain matrix given linear system (A,B) 
    # and cost function parameterized by (Q,R)
    
    # solve DARE:
    M = scipy.linalg.solve_discrete_are(A,B,Q,R)

    # K=(B'MB + R)^(-1)*(B'MA)
    return np.dot(scipy.linalg.inv(np.dot(np.dot(B.T,M),B)+R),(np.dot(np.dot(B.T,M),A)))


def run_linear_regression(env, agent, T=500, p=2, q=2,test_size=100, lag=4, plotVAR=False):
    X,_,_ = collect_episode(env, agent, T)
    data = np.transpose(X)
    train, test = data[0:len(data)-test_size], data[len(data)-test_size:]

    # train autoregression
    model = VAR(train)
    model_fit = model.fit(lag)
    window = model_fit.k_ar

    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = []
    # walk forward over time steps in test and make predictions
    for t in range(len(test)):
        obs = test[t]
        prediction = model_fit.forecast(history[len(history)-window:], 1)[0]
        print('predicted=%s, expected=%s' % (np.array2string(prediction), np.array2string(obs)))
        history.append(obs)
        predictions.append(prediction)
    error = mean_squared_error(test, predictions)
    print('Test MSE for lag=%d: %.3f' % (lag, error))
    # plot
    if plotVAR:
        plt.plot(test, color='blue')
        plt.plot(predictions, color='red')
        plt.show()
    return model_fit, data


def control_lqr(env, agent, model_fit, data, lag=4):
    B = np.array([[0],[0], [-.01], [-.01]])
    Q = np.diag((10., 1., 10., 1.))

    print(model_fit.coefs)

    K = lqr(model_fit.coefs[0], B, Q, 1)
    print("K=")
    print(K)

    obs = env.reset()
    action = agent.begin_episode(obs)
    for i in range(500):
        env.render()
        time.sleep(0.15) # slows down process to make it more visible

        # recompute K every 10 steps
        data = np.vstack([data, obs])
        if (i % 10 == 0):
            model_next = VAR(data)
            model_fit_next = model_next.fit(lag)
            K = lqr(model_fit_next.coefs[0], B, Q, 1)
            # print("K=")
            # print(K)

        action = get_control(K, obs)

        # Get the next action from the learner, given our new state.
        obs, reward, done, info = env.step(action)

        if done:
            print("Final episode: lasted {} timesteps, data: {}".format(i+1, obs))
            break


def main():
    monitor = False
    np.random.seed(seed=int(time.time()))
    # random_state = np.random.randint(2)

    env = gym.make("CartPole-v1")

    # env.seed(random_state)
    # np.random.seed(random_state)

    agent, episode_history = run_agent(env, verbose=False)   # Set verbose=False to greatly speed up the process.

    # close plot
    episode_history.close()

    # draw out the final policy and how it works
    if monitor:
        observation = env.reset()
        action = agent.begin_episode(observation)
        for t in range(1000):
            env.render()
            time.sleep(0.15) # slows down process to make it more visible

            # Get the next action from the learner, given our new state.
            observation, reward, done, info = env.step(action)
            action = agent.act_policy(observation) 

            if done:
                print("Final episode: lasted {} timesteps, data: {}".format(t+1, observation))
                break

    
    # avg_loss_vec = run_spectral_filtering(env, agent, 25, 500, num_trials=1)
    model_fit, data = run_linear_regression(env, agent, 4000)
    control_lqr(env, agent, model_fit, data)
    # control_lqr_finite_differences(env, agent)

    env.close()

if __name__ == "__main__":
    main()
