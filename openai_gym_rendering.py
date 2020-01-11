'''Rendering openai gym using matplotlib'''

import gym
import matplotlib.pyplot as plt

env = gym.make('Breakout-v0') # insert your favorite environment
render = lambda : plt.imshow(env.render(mode='rgb_array'))
env.reset()
render()




'''**********'''


import gym
from IPython import display
import matplotlib.pyplot as plt

env = gym.make('Breakout-v0')
env.reset()
for _ in range(100):
    plt.imshow(env.render(mode='rgb_array'))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    action = env.action_space.sample()
    env.step(action)
    plt.show()