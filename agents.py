from typing import Tuple

import gym
from keras.utils import to_categorical
import numpy as np

from models import create_MLP_model


class RandomAgent():
    """This agent generate several trials with random actions."""

    def __init__(
        self,
        env: gym.Env,
        n_trials: int,
        render: bool = False,
    ):
        """
        :param env: gym enviroment object.
        :param n_trials: number of trials to generate.
        :param render: whether to display the environment when generating trials.
        """
        self.env = env
        self.n_trials = n_trials
        self.render = render

        # Get the maximum number of step per trial
        self.n_max_steps = self.env.spec.max_episode_steps
        self.n_categories = self.env.action_space.n

    def play(self):
        """Generate trials and store each scores.

        :return: list of scores of all predicted trials.
        """
        scores = list()

        for trial in range(self.n_trials):
            observation = self.env.reset()
            score_trial = 0

            for step in range(self.n_max_steps):
                if self.render:
                    self.env.render()

                # Warning: `action` is related to the previous observation
                # Pick a random action (move left = 0, move right = 1)
                action = np.random.randint(0, self.n_categories)
                observation, reward, done, _ = self.env.step(action)
                score_trial += reward

                # Check wether the game is over or not
                if done:
                    break

            scores.append(score_trial)

        return np.array(scores)


class NaiveLearningAgent():
    """
    This agent generate several trials with random actions, and learn from trials with a score
    greater than a given threshold.
    """

    def __init__(
        self,
        env: gym.Env,
        min_score: int,
        n_training_trials: int,
        n_testing_trials: int = 100,
        training_render: bool = False,
        testing_render: bool = False,
    ):
        """
        :param env: gym enviroment object.
        :param min_score: minimum score to take into account an trial in the training data.
        :param n_training_trials: number of trials to gather training data.
        :param n_testing_trials: number of trials to evaluate the model (default = 100 trials).
        :param training_render: whether to display the environment when generating training trials.
        :param testing_render: whether to display the environment when playing test trials.
        """
        self.env = env
        self.min_score = min_score
        self.n_training_trials = n_training_trials
        self.n_testing_trials = n_testing_trials
        self.training_render = training_render
        self.testing_render = testing_render

        # Get number of input features (size of the observation space) and number of output
        # categories (size of the action space)
        self.n_features = self.env.observation_space.shape[0]
        self.n_categories = self.env.action_space.n
        # Get the maximum number of step per trial
        self.n_max_steps = env.spec.max_episode_steps

    def get_training_data(self) -> Tuple[np.array, np.array]:
        """
        Generate `n_training_trials` with `n_max_steps` and keep only trials' data whith a
        score greater than `min_score`.

        :return: observation of each steps and their related actions.
        """
        x_train = list()  # Store every observations of all trials
        y_train = list()  # Store every actions of all trials
        scores = list()   # Store every scores of all trials

        for trial in range(self.n_training_trials):
            observation = self.env.reset()

            score_trial = 0
            x_trial = list()  # Store every observations of the current trial
            y_trial = list()  # Store every actions of the current trial

            for step in range(self.n_max_steps):
                if self.training_render:
                    self.env.render()

                # Warning: `action` is related to the previous observation
                # Pick a random action (move left = 0, move right = 1)
                action = np.random.randint(0, self.n_categories)
                x_trial.append(observation)
                y_trial.append(action)

                observation, reward, done, _ = self.env.step(action)
                score_trial += reward

                # Check wether the game is over or not
                if done:
                    break

            if score_trial > self.min_score:
                x_train.extend(x_trial)
                y_train.extend(y_trial)
                scores.append(score_trial)

        print(f"Training score average: {np.mean(scores)}")
        print(f"Training score median: {np.median(scores)}")
        print(f"Number of training samples: {len(scores)}")
        print()

        return np.array(x_train), to_categorical(np.array(y_train))

    def play(self) -> np.array:
        """
        Generate training trials, create a model and make predictions for `n_testing_trials` of
        `n_max_steps`.

        :return: list of scores of all predicted trials.
        """
        x_train, y_train = self.get_training_data()

        model = create_MLP_model(self.n_features, self.n_categories)
        model.fit(x_train, y_train, epochs=5)

        scores = []
        for _ in range(self.n_testing_trials):
            observation = self.env.reset()

            score_trial = 0
            for step in range(self.n_max_steps):
                if self.testing_render:
                    self.env.render()

                # Get the model's prediction
                action = np.argmax(model.predict(observation.reshape(1, 4)))
                observation, reward, done, _ = self.env.step(action)
                score_trial += reward
                # Check wether the game is over or not
                if done:
                    break

            scores.append(score_trial)

        return np.array(scores)
