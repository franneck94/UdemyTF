# noqa
import gym
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class Agent:
    def __init__(self, env: gym.Env):
        pass

    def build_model(self) -> Sequential:
        pass

    def get_action(self, state: np.ndarray) -> int:
        pass

    def get_sample(self, num_episodes):
        pass

    def filter_episodes(self, rewards: list, episodes: list, percentile: float) -> tuple:
        pass

    def train(self, percentile: float, num_iterations: int, num_episodes: int) -> None:
        pass

    def play(self, num_episodes: int, render: bool = True) -> None:
        pass


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    # agent.train()
    # agent.play()
