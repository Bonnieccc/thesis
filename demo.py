import gym
import numpy as np
import random
import tensorflow as tf
# import cvxpy
import numpy as np

def discretize_range(lower_bound, upper_bound, num_bins):
    return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
    return np.digitize(x=value, bins=bins)

def build_state(observation, state_bins, max_bins):
    # Discretize the observation features and reduce them to a single integer.
    state = sum(
        discretize_value(feature, state_bins[i]) * ((max_bins + 1) ** i)
        for i, feature in enumerate(observation)
    )
    return state

def tabular_method():

	# Set learning parameters
	learning_rate = .8
	y = .95
	exploration_rate=0.5
	exploration_decay_rate=0.99


	env = gym.make('CartPole-v0')

	# Discretize the continuous state space for each of the 4 features.
	num_discretization_bins = 7
	state_bins = [
	    # Cart position.
	    discretize_range(-2.4, 2.4, num_discretization_bins),
	    # Cart velocity.
	    discretize_range(-3.0, 3.0, num_discretization_bins),
	    # Pole angle.
	    discretize_range(-0.5, 0.5, num_discretization_bins),
	    # Tip velocity.
	    discretize_range(-2.0, 2.0, num_discretization_bins)
	]

	# Create a clean Q-Table.
	num_actions = 2
	max_bins = max(len(bin) for bin in state_bins)
	num_states = (max_bins + 1) ** len(state_bins)
	Q = np.zeros(shape=(num_states, num_actions))

	#create lists to contain total rewards and steps per episode
	#jList = []
	rList = []

	num_episodes = 2000

	for ep in range(num_episodes):

		# exploration_rate *= exploration_decay_rate

		observation = env.reset()
		state = build_state(observation, state_bins, max_bins)
		print state
		action = np.argmax(Q[state])

        for t in range(500):
			# Perform the action and observe the new state.
			observation, reward, done, info = env.step(action)
			if done:
				break
			# print observation, reward

			# If the episode has ended prematurely, penalize the agent.
			# if done and timestep_index < max_timesteps_per_episode - 1:
			#     reward = -max_episodes_to_run

			# Get the next action from the learner, given our new state.
			next_state = build_state(observation, state_bins, max_bins)

	        # Exploration/exploitation: choose a random action or select the best one.
			# enable_exploration = (1 - exploration_rate) <= np.random.uniform(0, 1)
			# if enable_exploration:
			#     next_action = np.random.randint(0, num_actions)
			# else:
			next_action = np.argmax(Q[next_state])

			# Learn: update Q-Table based on current reward and future action.
			Q[state, action] += learning_rate * \
			    (reward + y * max(Q[next_state, :]) - Q[state, action])

			state = next_state
			action = next_action


	print "Score over time: " +  str(sum(rList)/num_episodes)
	print "Final Q-Table Values"
	print Q


tabular_method()


