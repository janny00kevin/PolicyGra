import time
import gym
import pygame
from gym.spaces import Discrete, Box

dim = [4] + [32]+ [2]
hidden_sizes=[32]
obs_dim = 4
n_acts = 2
sizes=[obs_dim]+hidden_sizes+[n_acts]
for j in range(len(sizes)-1):
    print(sizes[j])
# a = 3

env = gym.make('CartPole-v1', render_mode = 'human')
assert isinstance(env.observation_space, Box), \
    "This example only works for envs with continuous state spaces."
assert isinstance(env.action_space, Discrete), \
    "This example only works for envs with discrete action spaces."
# assert isinstance(a, float), \
#     "a is a integer"

obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n

print(obs_dim, n_acts)
print((dim))
# print(type(a))

# for _ in range(50):
#     env.reset()
#     for _ in range(1000):
#         env.render()
#         observation, reward, done, info, _ = env.step(env.action_space.sample())
#         if done:
#             break
#         time.sleep(0.02)
# env.close()