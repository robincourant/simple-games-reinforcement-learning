from typing import Tuple

import gym
from keras.utils import to_categorical
import numpy as np

from models import create_MLP_model


class NaiveLearningAgent():

    def __init__(
        self,
        env: gym.Env,
        min_score: int,
        n_training_episodes: int,
        n_training_steps: int,
        n_test_episodes: int,
        n_test_steps: int,
        training_render: bool = False,
        test_render: bool = False,
    ):
        """
        This agent generate several games with random actions, and learn from games with a score
        greater than a given threshold.

        :param env: gym enviroment object.
        :param min_score: minimum score to take into account an episode in the training data.
        :param n_training_episodes: number of simulations to gather training data.
        :param n_training_steps: number of steps in one episode to gather training data.
        :param n_test_episodes: number of simulations to evaluate the model.
        :param n_test_steps: number of steps in one episode to evaluate the model.
        :param training_render: whether to display the environment when generating training games.
        :param test_render: whether to display the environment when playing test games.
        """
        self.env = env
        self.min_score = min_score
        self.n_training_episodes = n_training_episodes
        self.n_training_steps = n_training_steps
        self.n_test_episodes = n_test_episodes
        self.n_test_steps = n_test_steps
        self.training_render = training_render
        self.test_render = test_render

        # Get number of input features (size of the observation space) and number of output
        # categories (size of the action space)
        self.n_features = self.env.observation_space.shape[0]
        self.n_categories = self.env.action_space.n

    def get_training_data(self) -> Tuple[np.array, np.array]:
        """
        Generate `n_training_episodes` with `n_training_steps` and keep only games' data whith a
        score greater than `min_score`.

        :return: observation of each steps and their related actions.
        """
        x_train = list()  # Store every observations of all episodes
        y_train = list()  # Store every actions of all episodes
        scores = list()   # Store every scores of all episodes

        for episode in range(self.n_training_episodes):
            observation = self.env.reset()

            score_episode = 0
            x_episode = list()  # Store every observations of the current episode
            y_episode = list()  # Store every actions of the current episode

            for step in range(self.n_training_steps):
                if self.training_render:
                    self.env.render()

                # Warning: `action` is related to the previous observation
                # Pick a random action (move left = 0, move right = 1)
                action = np.random.randint(0, 2)
                x_episode.append(observation)
                y_episode.append(action)

                observation, reward, done, _ = self.env.step(action)
                score_episode += reward

                # Check wether the game is over or not
                if done:
                    break

            if score_episode > self.min_score:
                x_train.extend(x_episode)
                y_train.extend(y_episode)
                scores.append(score_episode)

        print(f"Score average: {np.mean(scores)}")
        print(f"Score median: {np.median(scores)}")
        print(f"Number of training samples: {len(scores)}")

        return np.array(x_train), to_categorical(np.array(y_train))

    def predict(self):
        """
        Generate training games, create a model and make predictions for `n_test_episodes` of
        `n_test_steps`
        """
        x_train, y_train = self.get_training_data()

        model = create_MLP_model(self.n_features, self.n_categories)
        model.fit(x_train, y_train, epochs=5)

        scores = []
        for _ in range(self.n_test_episodes):
            observation = self.env.reset()

            score_episode = 0
            for step in range(self.n_test_steps):
                if self.test_render:
                    self.env.render()

                # Get the model's prediction
                action = np.argmax(model.predict(observation.reshape(1, 4)))
                observation, reward, done, _ = self.env.step(action)
                score_episode += reward
                # Check wether the game is over or not
                if done:
                    break

            scores.append(score_episode)

        print(f"Score average: {np.mean(scores)}")
        print(f"Score median: {np.median(scores)}")
