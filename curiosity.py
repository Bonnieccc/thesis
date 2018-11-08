# Bin state/action space into N discrete bins
# Randomly move agent around in the space for T steps. 
# Get a frequency distribution over the number of times you visit bins.
	# Transform this into a probability distribution
# Compute the entropy measure over the distribution 
# Compute gradient of entropy with respect to p1...pn
	# This is the reward function for the iteration
	# Find new distribution p_t+1 that maximizes this reward
	# This distribution becomes new policy

# At the end, return the average policy over all iterations (i.e. average over all pi)



# Later: update function weighted to smooth changes
# pt* = p_t-1*(1 - 1/t) + 1/t(pt)

# want to examne the distribution achieved over the state/action space

import numpy as np
import gym
import time
from gym.spaces import prng
import tensorflow as tf
import tensorflow.contrib.slim as slim


def discretize_range(lower_bound, upper_bound, num_bins):
	return np.linspace(lower_bound, upper_bound, num_bins + 1)[1:-1]

def discretize_value(value, bins):
	return np.asscalar(np.digitize(x=value, bins=bins))

def discretize_state(observation, state_bins):
    # Discretize the observation features and reduce them to a single list.
    state = []
    for i, feature in enumerate(observation):
        state.append(discretize_value(feature, state_bins[i]))
    return state

# map negative actions to 0, positive to 1
def discretize_action(action):
	if np.sign(action) == 1:
		return 1
	return 0

class Agent():
    def __init__(self, lr, s_size,a_size,h_size):
        #These lines established the feed-forward part of the network. The agent takes a state and produces an action.
        self.state_in= tf.placeholder(shape=[None,s_size],dtype=tf.float32)
        hidden = slim.fully_connected(self.state_in,h_size,biases_initializer=None,activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,a_size,activation_fn=tf.nn.softmax,biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        #The next six lines establish the training proceedure. We feed the reward and chosen action into the network
        #to compute the loss, and use it to update the network.
        self.reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None],dtype=tf.int32)
        
        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)
        
        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
            self.gradient_holders.append(placeholder)
        
        self.gradients = tf.gradients(self.loss,tvars)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders,tvars))

gamma = 0.99

def discount_rewards(r):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def is_goal_reached(rewards, goal_consecutive_episodes, goal_reward):
    avg = np.mean(rewards[-goal_consecutive_episodes:])
    return avg >= goal_reward and len(rewards) > goal_consecutive_episodes

def train_agent(env, reward, state_bins):
    tf.reset_default_graph() #Clear the Tensorflow graph.

    agent = Agent(lr=1e-2, s_size=2, a_size=2, h_size=8) #Load the agent.

    total_episodes = 2000 # Set total number of episodes to train agent on.
    episode_steps = 999
    update_frequency = 5

    init = tf.global_variables_initializer()
    avg_history = []
    # Launch the tensorflow graph
    with tf.Session() as sess:
        sess.run(init)
        i = 0

        gradBuffer = sess.run(tf.trainable_variables())
        for ix,grad in enumerate(gradBuffer):
            gradBuffer[ix] = grad * 0
            
        episode_rewards = []
        episode_lengths = []
        while i < total_episodes:
            state = env.reset()
            running_reward = 0
            ep_history = []
            for j in range(episode_steps):
                #Probabilistically pick an action given our network outputs.
                a_dist = sess.run(agent.output,feed_dict={agent.state_in:[state]})
                a = np.random.choice(a_dist[0],p=a_dist[0])
                a = np.argmax(a_dist == a)

                next_state, r, done, info = env.step([a])
                ep_history.append([state,a,r,next_state])
                state = next_state
                running_reward += reward[discretize_state(state, state_bins)]
                if done:
                    #Update the network.
                    ep_history = np.array(ep_history)
                    ep_history[:,2] = discount_rewards(ep_history[:,2])
                    feed_dict = {agent.reward_holder:ep_history[:,2],
                                agent.action_holder:ep_history[:,1], 
                                agent.state_in:np.vstack(ep_history[:,0])}
                    grads = sess.run(agent.gradients, feed_dict=feed_dict)
                    for idx,grad in enumerate(grads):
                        gradBuffer[idx] += grad

                    if i % update_frequency == 0 and i != 0:
                        feed_dict= dictionary = dict(zip(agent.gradient_holders, gradBuffer))
                        _ = sess.run(agent.update_batch, feed_dict=feed_dict)
                        for ix,grad in enumerate(gradBuffer):
                            gradBuffer[ix] = grad * 0
                    
                    episode_rewards.append(running_reward)
                    episode_lengths.append(j)
                    break

            #Update our running tally of scores.
            if i % 100 == 0:
                avg = np.mean(episode_rewards[-100:])
                print(np.mean(avg))
                equal = [x for x in avg_history[-3:] if x == avg]
                print(avg_history)
                print(equal)
                if len(equal) == 3:
                    return agent
                avg_history.append(avg)
            i += 1

            # # return if the last 1000 have adequate avg reward.
            # if is_goal_reached(episode_rewards, 1000, 999):
            #     return agent

    return agent

# act according to the agent's learned policy.
def execute_policy(env, agent, steps, state_bins):
    num_actions = 2
    num_states = [(len(state_bins[0]) + 1),
                  (len(state_bins[1]) + 1)]

    p = np.zeros(shape=(num_states[0],
                        num_states[1]))

    obs = env.reset()
    with tf.Session() as sess:
    	for step in range(steps):
    		# Probabilistically pick an action given our network outputs.
    	    a_dist = sess.run(agent.output,feed_dict={agent.state_in:[obs]})
    	    a = np.random.choice(a_dist[0],p=a_dist[0])
    	    a = np.argmax(a_dist == a)

    	    obs, reward, done, info = env.step([a])
    	    p[discretize_state(obs, state_bins)] += 1

    p = np.reshape(p/float(steps), -1)
    return p

# TODO: Is this right????
def grad_ent(pt):
    return -np.log(pt) - 1

def main():
    nx = 15
    nv = 15
    S = nx*nv
    state_bins = [
        # Cart position.
        discretize_range(-1.2, 0.6, nx), 
        # Cart velocity.
        discretize_range(-0.07, 0.07, nv)
    ]

    num_actions = 2
    num_states = [(len(state_bins[0]) + 1),
    			  (len(state_bins[1]) + 1)]


    # Make environment.
    env = gym.make("MountainCarContinuous-v0")
    env.seed(int(time.time())) # seed environment
    prng.seed(int(time.time())) # seed action space

    steps = 1000

    T = 1000
    p = np.ones(shape=(num_states[0],
                       num_states[1]))/2.
    reward = np.ones(shape=(num_states[0],
                            num_states[1]))
    for i in range(T):
        # print(reward)
        reward = grad_ent(p)
        # Learn policy that maximizes current reward function.
        agent = train_agent(env, reward, state_bins)
        # Get next distribution p by executing pi.
        p = execute_policy(env, agent, steps, state_bins)
        print("p_t = ")
        print(p)

# Find pt
# Compute gradient of entry of pt —-> take as reward function
# Perform policy gradient to find optimal policy pi given the reward function
# Run it around in the environment to get the next pt+1


if __name__ == "__main__":
    main()





