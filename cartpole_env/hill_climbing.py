import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode="human")

num_episodes = 1000  
learning_rate = 0.1  
noise_scale = 0.1  
gamma = 0.99  # discount

weights = np.random.randn(4) * 0.1 

def policy(state, weights):
    return 1 if np.dot(state, weights) > 0 else 0

def run_episode(env, weights):
    state = env.reset()[0]
    total_reward = 0
    done = False
    while not done:
        action = policy(state, weights)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward

rewards = []  
best_weights = weights  
best_reward = 0  
count = 0

for episode in range(num_episodes):
    new_weights = best_weights + noise_scale * np.random.randn(4)
    
    total_reward = run_episode(env, new_weights)
    rewards.append(total_reward)

    if total_reward > best_reward:
        best_reward = total_reward
        best_weights = new_weights

    noise_scale *= 0.99

    if (episode + 1) % 10 == 0:
        print(f"Episode {episode + 1}/{num_episodes}, Best Reward: {best_reward}")
        if best_reward > 200:
            count += 1
        if count > 5:
            break

plt.plot(rewards)
plt.title("Total reward over episodes - Hill Climbing")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

print("Testing the best policy...")
state = env.reset()[0]
done = False
total_reward = 0
while not done:
    env.render()
    action = policy(state, best_weights)
    state, reward, done, _, _ = env.step(action)
    total_reward += reward
env.close()

print(f"Test Reward: {total_reward}")
