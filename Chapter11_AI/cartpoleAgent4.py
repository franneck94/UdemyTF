import collections
import random

import gym
import numpy as np

from cartpoleDqn4 import *


class Agent:
    def __init__(self, env):
        # DQN Env Variables
        self.env = env
        self.actions = self.env.action_space.n
        self.observations = self.env.observation_space.shape
        # DQN Agent Variables
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.replay_buffer_size = 50000
        self.train_start = 1000
        self.memory = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-3
        self.model = DQN(self.state_shape, self.actions, self.learning_rate)
        self.batch_size = 32

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.model.predict(state))

    def train(self, num_episodes):
        best_total_reward = 0.0
        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, (1, state.shape[0]))

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, next_state.shape[0]))
                if done and total_reward < 499:
                    reward = -100

                self.remember(state, action, reward, next_state, done)
                self.replay()

                total_reward += reward
                state = next_state

                if done:
                    if total_reward != 500:
                        total_reward += 100

                    if total_reward > best_total_reward:
                        best_total_reward = total_reward
                        self.model.save_model("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/models/dqn.h5")

                    print(
                        "Episode: ",
                        episode + 1,
                        " Total Reward: ",
                        total_reward,
                        " Epsilon:",
                        self.epsilon,
                    )
                    break

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def replay(self):
        if len(self.memory) < self.train_start:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, states_next, dones = zip(*minibatch)

        states = np.concatenate(states)
        states_next = np.concatenate(states_next)

        q_values = self.model.predict(states)
        q_values_next = self.model.predict(states_next)

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a] = rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * np.max(q_values_next[i])

        self.model.train(states, q_values)

    def play(self, num_episodes, render=True):
        self.model.load_model("C:/Users/Jan/Dropbox/_Programmieren/UdemyTF/models/dqn.h5")
        for episode in range(num_episodes):
            state = self.env.reset()
            state = np.reshape(state, (1, state.shape[0]))
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, next_state.shape[0]))
                state = next_state
                if render:
                    self.env.render()
                if done:
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(num_episodes=150)
    input("Play?")
    agent.play(num_episodes=20, render=True)
