import gym
import tensorflow as tf
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Activation, Reshape
from keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.agents.dqn import DQNAgent
env = gym.make('CartPole-v1')
env.max_episode_steps = 500
num_actions = env.action_space.n
# Creating out simple NN model here 
observation = Input(shape=(1, ) + env.observation_space.shape)
x = Dense(16, activation='relu')(observation)
x = Dense(16, activation='relu')(x)
x = Dense(16, activation='relu')(x)
output = Dense(num_actions, activation='linear')(x) 
output = Reshape((num_actions, ))(output)
model = Model(inputs=observation, outputs=output)
print(model.summary())