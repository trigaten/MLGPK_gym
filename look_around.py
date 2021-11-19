"""Agent for the mlg_wb gym which just stands still and looks around."""

import gym

from mlg_wb_specs import MLGWB

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

abs_MLG = MLGWB()
abs_MLG.register()
env = gym.make("MLGWB-v0")

obs  = env.reset()
done = False
net_reward = 0

while not done:

    env.render()

    action = env.action_space.noop()
    
    # action['back'] = 0
    action['forward'] = 1
    action["camera"] = [0.3, 0.7]
    action["place"] = 1
    # action['jump'] = 1
    # action['attack'] = 1

    obs, reward, done, info = env.step(
        action)

    net_reward += reward
    print("Total reward: ", net_reward)