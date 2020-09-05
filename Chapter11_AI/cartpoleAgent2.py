import gym
import numpy as np


class Agent:
    def __init__(self, env):
        # DQN Env Variables
        self.env = env
        self.actions = self.env.action_space.n
        self.observations = self.env.observation_space.shape

    def get_action(self):
        return np.random.randint(self.actions)

    def train(self):
        pass

    def remember(self):
        pass

    def replay(self):
        pass

    def play(self, num_episodes, render=True):
        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                action = self.get_action()
                _, _, done, _ = self.env.step(action)
                if render:
                    self.env.render()
                if done:
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    # agent.train()
    input("Play?")
    agent.play(num_episodes=100, render=True)
