# noqa
import gym
import numpy as np
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


class Agent:
    def __init__(self, env):
        self.env = env
        self.num_obersvations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(units=100, input_dim=self.num_obersvations))
        model.add(Activation("relu"))
        model.add(Dense(units=self.num_actions))
        model.add(Activation("softmax"))
        model.summary()
        model.compile(
            loss="categorical_crossentropy",
            optimizer="Adam",
            metrics=["accuracy"]
        )
        return model

    def get_action(self, state):
        state = state.reshape(1, -1)
        action_prob = self.model(state).numpy()[0]
        action = np.random.choice(self.num_actions, p=action_prob)
        return action

    def get_samples(self, num_episodes):
        rewards = [0.0 for _ in range(num_episodes)]
        episodes = [[] for _ in range(num_episodes)]

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                action = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                episodes[episode].append((state, action))
                state = new_state
                if done:
                    rewards[episode] = total_reward
                    break
        return rewards, episodes

    def filter_episodes(self, rewards, episodes, percentile):
        reward_bound = np.percentile(rewards, percentile)
        x_train, y_train = [], []
        for reward, episode in zip(rewards, episodes):
            if reward >= reward_bound:
                observations = [step[0] for step in episode]
                actions = [step[1] for step in episode]
                x_train.extend(observations)
                y_train.extend(actions)
        x_train = np.array(x_train)
        y_train = to_categorical(y_train, num_classes=self.num_actions)
        return x_train, y_train, reward_bound

    def train(self, percentile, num_iterations, num_episodes):
        for _ in range(num_iterations):
            rewards, episodes = self.get_samples(num_episodes)
            x_train, y_train, reward_bound = self.filter_episodes(rewards, episodes, percentile)
            self.model.fit(x=x_train, y=y_train, verbose=0)
            reward_mean = np.mean(rewards)
            print(f"Reward mean: {reward_mean}, reward bound: {reward_bound}")
            if reward_mean > 450:
                break

    def play(self, num_episodes, render=True):
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0.0

            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
                if done:
                    print(f"Total reward: {total_reward} in episode {episode + 1}")
                    break


if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    agent.train(percentile=70.0, num_iterations=15, num_episodes=100)
    input()
    agent.play(num_episodes=10)
