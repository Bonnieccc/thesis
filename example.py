import gym
import time
import numpy as np

env = gym.make('Pendulum-v0')

highscore = 0
obs = env.reset()
for i_episode in range(1): # run 20 episodes
  observation = env.step([0])
  print(observation)
  print(env.env.state)
  env.env.state = [np.pi, 0]
  print(env.env._get_obs())

  # print(np.arcsin( -0.01466774))
  # -0.99989242, -0.01466774,  0.02598429]

  env.render()
  time.sleep(10)
  # time.sleep(5)
  # points = 0 # keep track of the reward each episode
  # while True: # run until episode is done
  #   env.render()
  #   action = 1 if observation[2] > 0 else 0 # if angle if positive, move right. if angle is negative, move left
  #   observation, reward, done, info = env.step(action)
  #   points += reward
  #   if done:
  #     if points > highscore: # record high score
  #       highscore = points
  #       break
env.close()


# env = gym.make('MountainCarContinuous-v0')
# obs = env.reset()
# for i_episode in range(1): # run 20 episodes
#   env.env.state = [-0.50, 0]
#   env.render()
#   time.sleep(10)

# env.close()