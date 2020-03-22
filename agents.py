from typing import Tuple

import gym
from keras.utils import to_categorical
import numpy as np

from models import create_MLP_classifier


class Agent:

    def __init__(
        self,
        env: gym.Env,
    ):
        """
        :param env: gym enviroment object.
        """
        self.env = env
        # Get the maximum number of step per trial and the size of the action and observation space
        self.n_max_steps = self.env.spec.max_episode_steps
        self.action_space_size = self.env.action_space.n
        self.observation_space_size = self.env.observation_space.shape


class RandomAgent(Agent):
    """Pick random actions."""

    def play(
        self,
        n_episodes: int = 100,
        render: bool = False,
    ):
        """Generate `n_episodes` trials and return every scores.

        : param n_episodes: number of trials to generate (default: 100 trials).
        : param render: whether to display the environment when generating trials default: False).
        : return: list of scores of all predicted trials.
        """
        scores = list()
        for trial in range(n_episodes):
            observation = self.env.reset()
            score_trial = 0

            for step in range(self.n_max_steps):
                if render:
                    self.env.render()

                # Warning: `action` is related to the previous observation
                # Pick a random action (move left = 0, move right = 1)
                action = np.random.randint(0, self.action_space_size)
                observation, reward, done, _ = self.env.step(action)
                score_trial += reward

                # Check wether the game is over or not
                if done:
                    break

            scores.append(score_trial)

        return np.array(scores)


class NaiveLearningAgent(Agent):
    """
    Generate several trials with random actions, and learn from trials with a score
    greater than a given threshold.
    """

    def __init__(
        self,
        env: gym.Env,
    ):
        super().__init__(env)
        self.model = create_MLP_classifier(self.observation_space_size[0], self.action_space_size)

    def get_training_data(
        self,
        min_score,
        n_training_episodes,
        training_render,
    ) -> Tuple[np.array, np.array]:
        """
        Generate `n_training_episodes` with `n_max_steps` and keep only trials' data whith a
        score greater than `min_score`.

        : return: observation of each steps and their related actions.
        """
        x_train = list()  # Store every observations of all trials
        y_train = list()  # Store every actions of all trials
        scores = list()   # Store every scores of all trials

        for trial in range(n_training_episodes):
            observation = self.env.reset()

            score_trial = 0
            x_trial = list()  # Store every observations of the current trial
            y_trial = list()  # Store every actions of the current trial

            for step in range(self.n_max_steps):
                if training_render:
                    self.env.render()

                # Warning: `action` is related to the previous observation
                # Pick a random action (move left = 0, move right = 1)
                action = np.random.randint(0, self.action_space_size)
                x_trial.append(observation)
                y_trial.append(action)

                observation, reward, done, _ = self.env.step(action)
                score_trial += reward

                # Check wether the game is over or not
                if done:
                    break

            if score_trial > min_score:
                x_train.extend(x_trial)
                y_train.extend(y_trial)
                scores.append(score_trial)

        print(f"Training score average: {np.mean(scores)}")
        print(f"Training score median: {np.median(scores)}")
        print(f"Number of training samples: {len(scores)}")
        print()

        return np.array(x_train), to_categorical(np.array(y_train))

    def play(
        self,
        min_score: int,
        n_training_episodes: int,
        n_testing_episodes: int = 100,
        training_render: bool = False,
        testing_render: bool = False,
    ) -> np.array:
        """
        Generate training trials, create a model and make predictions for `n_testing_episodes` of
        `n_max_steps`.

        : param min_score: minimum score to take into account an trial in the training data.
        : param n_training_episodes: number of trials to gather training data.
        : param n_testing_episodes: number of trials to evaluate the model(default: 100 trials).
        : param training_render: whether to display the environment when generating training trials.
        : param testing_render: whether to display the environment when playing test trials.
        : return: list of scores of all predicted trials.
        """
        x_train, y_train = self.get_training_data(min_score, n_training_episodes, training_render)
        self.model.fit(x_train, y_train, epochs=5)

        scores = []
        for _ in range(n_testing_episodes):
            observation = self.env.reset()

            score_trial = 0
            for step in range(self.n_max_steps):
                if testing_render:
                    self.env.render()

                # Get the model's prediction
                action = np.argmax(self.model.predict(observation.reshape(1, 4)))
                observation, reward, done, _ = self.env.step(action)
                score_trial += reward
                # Check wether the game is over or not
                if done:
                    break

            scores.append(score_trial)

        return np.array(scores)
