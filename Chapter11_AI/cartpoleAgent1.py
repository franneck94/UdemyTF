# noqa
import gym
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class Agent:
    def __init__(self, env):
        pass

    def build_model(self):
        pass

    def get_action(self, state):
        pass

    def get_sample(self, num_episodes):
        pass

    def filter_episodes(self, rewards, episodes, percentile):
        pass

    def train(self, percentile, num_iterations, num_episodes):
        pass

    def play(self, num_episodes, render=True):
        pass


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    # agent.train()
    # agent.play()
