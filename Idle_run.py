import gym
import highway_env
from matplotlib import pyplot as plt
env = gym.make('highway-v0')
env.config["duration"] = 50
env.config["simulation_frequency"] = 15
env.config["show_trajectories"] = True
env.config['real_time_rendering'] = True
env.config['frequency'] = 20
env.config['manual_control'] = True
obs = env.reset()
for _ in range(30):
    action = env.action_type.actions_indexes["IDLE"]
    obs, reward, done, info = env.step(action)
    env.render()
plt.imshow(env.render(mode="rgb_array"))
plt.show()
env.close()

