"""
https://stackoverflow.com/questions/73916260/how-can-i-render-openai-gym-in-windows-python3cartpole
"""

import gym
from tqdm import tqdm

n=500

env = gym.make("CartPole-v1", render_mode="human")
env.action_space.seed(82)

observation, info = env.reset(seed=82)

for _ in tqdm(range(n)):
    action = env.action_space.sample()
    observation,reward, terminated, truncated, info = env.step(action)
    print("info : ",info);
    
    if terminated or truncated:
        observation, info = env.reset()
        
env.close()