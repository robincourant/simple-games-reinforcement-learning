import pickle
import random
from time import time, sleep
from typing import Any, List, Tuple

import gym
from keras.models import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
import numpy as np

from models import create_mlp_model, save_model


class Agent:

    def __init__(
        self,
        env: gym.Env,
        reward_threshold: int = -np.inf,
    ):
        """
        :param env: gym enviroment object.
        :param reward_threshold: score threshold to reach.
        """
        self.env = env
        self.env_name = self.env.spec.id
        self.reward_threshold = reward_threshold

        # Get the maximum number of step per trial and the size of the action and state space
        self.n_max_steps = self.env.spec.max_episode_steps
        self.action_space_size = self.env.action_space.n
        self.state_space_size = self.env.observation_space.shape


class KeyboardAgent(Agent):
    """Keyboard policy. Press 'r' to restart a trial and spacebar to pause it."""

    # Up: 273 -> 65362 / Down: 274 -> 65364 / Left: 276 -> 65361 / Right: 275 -> 65363
    # Spacebar: 27 -> 32 / r: 114 -> 114
    # NanoNotes: http://en.qi-hardware.com/wiki/Key_codes
    # keysym: http://www.tcl.tk/man/tcl8.4/TkCmd/keysyms.htm
    keysyms_to_NanoNotes = {65362: 273, 65364: 274, 65363: 275, 65361: 276, 32: 27, 114: 114}

    def __init__(self, *args, keys_to_action=None):
        super().__init__(*args)

        self.action = 0
        self.restart = False
        self.pause = False

        if not keys_to_action:
            try:
                self.keys_to_action = self.env.get_keys_to_action()
            except AttributeError:
                pass

            try:
                self.keys_to_action = self.env.unwrapped.get_keys_to_action()
            except AttributeError:
                print(f"{self.env.spec.id} does not have explicit key to action mapping, "
                      "please specify one manually")
        else:
            self.keys_to_action = keys_to_action

        self.NO_ACTION_KEY = self.keys_to_action[()] if () in self.keys_to_action else 0
        self.RESTART_KEY = 114
        self.PAUSE_KEY = 27

    def key_press(
        self,
        symbol: int,
        modifiers: int,
    ):
        """Update `action` variable when a key is pressed.

        :param symbol: key symbol pressed.
        :param modifiers: bitwise combination of the key modifiers active.
        """
        key = KeyboardAgent.keysyms_to_NanoNotes.get(symbol)
        if symbol:
            if key == self.RESTART_KEY:
                self.restart = not self.restart
            if key == self.PAUSE_KEY:
                self.pause = not self.pause
            if (key, ) in self.keys_to_action:
                self.action = self.keys_to_action[(key, )]

    def key_release(
        self,
        symbol: int,
        modifiers: int,
    ):
        """Update `action` variable when a key is released.

        :param symbol: key symbol pressed.
        :param modifiers: bitwise combination of the key modifiers active.
        """
        key = KeyboardAgent.keysyms_to_NanoNotes.get(symbol)
        if symbol:
            if (key, ) in self.keys_to_action:
                if self.action == self.keys_to_action[(key, )]:
                    self.action = self.NO_ACTION_KEY

    def play(
        self,
        n_episodes: int = 1,
    ) -> np.array:
        """Generate `n_episodes` trials and return every scores.

        :param n_episodes: number of trials to generate (default: 1 trial).
        :return: list of scores of all predicted trials.
        """
        self.env.reset()
        self.env.render()
        self.env.unwrapped.viewer.window.on_key_press = self.key_press
        self.env.unwrapped.viewer.window.on_key_release = self.key_release

        scores = list()
        k_episode = 0
        while k_episode < n_episodes:
            self.env.reset()
            self.restart = False
            score_trial = 0

            k_step = 0
            while k_step < self.n_max_steps:
                self.env.render()
                # Warning: `action` is related to the previous state
                _, reward, done, _ = self.env.step(self.action)
                score_trial += reward
                k_step += 1
                # Check whether the user wants to pause the trial
                while self.pause:
                    self.env.render()
                    self.pause
                # Check whether the user wants to restart the trial
                if self.restart:
                    break
                # Check whether the game is over or not
                if done:
                    break

                sleep(0.01)

            scores.append(score_trial)
            k_episode += 1

        return np.array(scores)


class RandomAgent(Agent):
    """Random policy."""

    def play(
        self,
        n_episodes: int = 100,
        render: bool = False,
    ) -> np.array:
        """Generate `n_episodes` trials and return every scores.

        :param n_episodes: number of trials to generate (default: 100 trials).
        :param render: whether to display the environment when generating trials default: False).
        :return: list of scores of all predicted trials.
        """
        scores = list()
        for trial in range(n_episodes):
            self.env.reset()
            score_trial = 0

            for step in range(self.n_max_steps):
                if render:
                    self.env.render()

                # Warning: `action` is related to the previous state
                action = np.random.randint(0, self.action_space_size)
                _, reward, done, _ = self.env.step(action)
                score_trial += reward

                # Check whether the game is over or not
                if done:
                    break

            scores.append(score_trial)

        return np.array(scores)


class NaiveLearningAgent(Agent):
    """Policy learnt from best games of a random agent"""

    def __init__(self, *args):
        super().__init__(*args)

        self.save = False

        file_path = "/".join(["models", self.env_name, "naive_learning_agent.pkl"])
        try:
            self.model = pickle.load(open(file_path, "rb"))
            print("Model loaded.")
        except FileNotFoundError:
            print("No existing model. Training one ...")
            t0 = time()
            self.model = self.train_model()
            print(f"Model trained in {time() - t0:.2f}s.")
            if self.save:
                save_model(self.model, file_path)

    def get_training_data(
        self,
    ) -> Tuple[np.array, np.array]:
        """
        Generate `n_training_episodes` with `n_max_steps` and keep only trials' data whith a
        score greater than `min_score`.

        : return: state of each steps and their related actions.
        """
        # minimum score to take into account an trial in the training data
        min_score = 70
        # number of trials to gather training data
        n_episodes = int(10e3)

        x_train = list()  # Store every states of all trials
        y_train = list()  # Store every actions of all trials
        scores = list()   # Store every scores of all trials

        for _ in range(n_episodes):
            state = self.env.reset()

            score_trial = 0
            x_trial = list()  # Store every states of the current trial
            y_trial = list()  # Store every actions of the current trial

            for step in range(self.n_max_steps):
                # Warning: `action` is related to the previous state
                # Pick a random action (move left = 0, move right = 1)
                action = np.random.randint(0, self.action_space_size)
                new_state, reward, done, _ = self.env.step(action)

                x_trial.append(state)
                y_trial.append(action)
                score_trial += reward
                state = new_state

                # Check whether the game is over or not
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

    def train_model(
        self
    ) -> Sequential:
        """Gather training data and train a basic MLP model.

        :return: trained model.
        """
        x_train, y_train = self.get_training_data()

        model = create_mlp_model(
            n_features=self.state_space_size[0],
            n_categories=self.action_space_size,
            loss="categorical_crossentropy",
            proba_output=True,
        )
        model.fit(x_train, y_train, epochs=5)

        return model

    def play(
        self,
        n_episodes: int = 100,
        render: bool = False,
    ) -> np.array:
        """Generate `n_episodes` trials and return every scores.

        :param n_episodes: number of trials to generate (default: 100 trials).
        :param render: whether to display the environment when generating trials default: False).
        :return: list of scores of all predicted trials.
        """
        scores = []
        for _ in range(n_episodes):
            state = self.env.reset()

            score_trial = 0
            for step in range(self.n_max_steps):
                if render:
                    self.env.render()

                # Get the model's prediction
                action = np.argmax(self.model.predict(
                    state.reshape(1, self.state_space_size[0])))
                state, reward, done, _ = self.env.step(action)
                score_trial += reward
                # Check whether the game is over or not
                if done:
                    break

            scores.append(score_trial)

        return np.array(scores)


class DeepQLearningAgent(Agent):
    """Policy learnt throught deep Q-learning."""

    def __init__(self, *args):
        super().__init__(*args)

        self.replay_memory: List[List[Any]] = list()
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.tau = .125
        self.batch_size = 32
        self.training_render = False
        self.max_training_episode = 500
        self.save = False

        file_path = "/".join(["models", self.env_name, "deep_Q_learning_agent.pkl"])
        try:
            self.model = pickle.load(open(file_path, "rb"))
            print("Model loaded.")
        except FileNotFoundError:
            print("No existing model. Training one ...")
            t0 = time()
            self.model = create_mlp_model(
                n_features=self.state_space_size[0],
                n_categories=self.action_space_size,
                batch_size=32,
                loss="mean_squared_error",
                optimizer=Adam(lr=self.learning_rate),
                proba_output=False,
            )
            self.target_model = create_mlp_model(
                n_features=self.state_space_size[0],
                n_categories=self.action_space_size,
                batch_size=32,
                loss="mean_squared_error",
                optimizer=Adam(lr=self.learning_rate),
                proba_output=False,
            )
            self.model = self.train_model()
            print(f"Model trained in {time() - t0:.2f}s.")
            if self.save:
                save_model(self.model, file_path)

    def get_action(
        self,
        state: np.array,
    ) -> int:
        """
        Select a random action whith probability `epsilon` or select the best
        action provided by `model`.

        :param state: agent's observation of the current environment.
        :return: selected action.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def store_transition(
        self,
        state: np.array,
        action: int,
        reward: int,
        new_state: np.array,
        done: bool,
    ):
        """Store the agent’s experiences in replay memory.

        :param state: agent's observation of the current environment.
        :param action: an action provided by the agent.
        :param reward: amount of reward returned after previous action.
        :param new_state: agent's observation of the next environment.
        :param done: whether the episode has ended.
        """
        self.replay_memory.append([state, action, reward, new_state, done])

    def replay(
        self
    ):
        """Experience replay: train model from `batch_size` random previous agent’s experiences."""
        # Check there is enough sample to feed the model
        if len(self.replay_memory) < self.batch_size:
            return

        # Pick `batch_size` random samples from `replay_memory`
        samples = np.array(random.sample(self.replay_memory, self.batch_size))
        states = np.array([it[0] for it in samples]).reshape(self.batch_size, -1)
        new_states = np.array([it[3] for it in samples]).reshape(self.batch_size, -1)
        dones = np.array([it[4] for it in samples]).reshape(self.batch_size, -1)
        rewards = np.array([it[2] for it in samples]).reshape(self.batch_size, -1)
        actions = np.array([it[1] for it in samples]).reshape(self.batch_size, -1)
        # Find best actions predicted by current Q-values
        Q_values = self.target_model.predict(states)
        Q_next = np.amax(self.target_model.predict(new_states), axis=1).reshape(self.batch_size, -1)
        # Terminal states' rewards are not changed, but cumulative dicounted reward is added to
        # non-terminal states' rewards, and update Q-values
        new_rewards = (rewards + ((1 - dones) * Q_next * self.gamma)).reshape(self.batch_size, -1)
        np.put_along_axis(Q_values, actions.reshape(self.batch_size, -1), new_rewards, axis=1)

        self.model.fit(states, Q_values, epochs=1, verbose=0)

    def update_weights(
        self
    ):
        """Update weights of the main main with weights of the target model."""
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def train_model(
        self
    ) -> Sequential:
        """Train the deep Q-network and save it.

        :return: trained model.
        """
        for trial in range(self.max_training_episode):
            t0 = time()
            score_trial = 0
            state = self.env.reset().reshape(1, self.state_space_size[0])

            for step in range(self.n_max_steps):
                if self.training_render:
                    self.env.render()
                # Select a random action or the best prediction of `model`
                action = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = new_state.reshape(1, self.state_space_size[0])
                self.store_transition(state, action, reward, new_state, done)
                self.replay()
                self.update_weights()

                state = new_state
                score_trial += reward

                if done:
                    break

            # Check whether the trial is completed of not
            if score_trial <= self.reward_threshold:
                print(f"Failed to complete in trial {trial} "
                      f"(score: {score_trial}, time: {time() - t0:.2f}s)")
            else:
                print(f"Completed in {trial+1} trials "
                      f"(score: {score_trial}, time: {time() - t0:.2f}s)")
                return self.model

    def play(
        self,
        n_episodes: int = 100,
        render: bool = True,
    ) -> np.array:
        """
        Generate `n_episodes` trials and return every scores.

        :param n_episodes: number of trials to generate (default: 100 trials).
        :param render: whether to display the environment when generating trials default: False).
        :return: list of scores of all predicted trials.
        """
        scores = []
        for _ in range(n_episodes):
            state = self.env.reset()

            score_trial = 0
            for step in range(self.n_max_steps):
                if render:
                    self.env.render()

                # Get the model's prediction
                action = np.argmax(self.model.predict(
                    state.reshape(1, self.state_space_size[0])))
                state, reward, done, _ = self.env.step(action)
                score_trial += reward
                # Check whether the game is over or not
                if done:
                    break

            scores.append(score_trial)
            print(f"score_trial: {score_trial}")

        return np.array(scores)
