import random
from time import time, sleep
from typing import Any, List, Tuple

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

from models import (
    create_actor_critic_mlp,
    create_random_mlp,
    create_small_mlp,
    save_model,
)


class BaseAgent:
    """Basic agent with specified environment (+ action and state space features) and score threshold."""

    def __init__(
        self,
        env: gym.Env,
        reward_threshold: int = -np.inf,
        model_path: str = None,
    ):
        """
        :param env: gym enviroment object.
        :param reward_threshold: score threshold to reach.
        """
        self.env = env
        self.env_name = self.env.spec.id
        self.reward_threshold = reward_threshold
        self.model_path = model_path

        # Get the maximum number of step per trial
        self.n_max_steps = self.env.spec.max_episode_steps
        # Get the size of the action
        if isinstance(self.env.action_space, gym.spaces.discrete.Discrete):
            self.action_space_size = self.env.action_space.n
        if isinstance(self.env.action_space, gym.spaces.box.Box):
            self.action_space_size = self.env.action_space.shape[0]
        # Get the state of the action
        self.state_space_size = self.env.observation_space.shape[0]


class SmartAgent(BaseAgent):
    """
    Agent that can play a game with a method `play_next_step` and
    a model that can predict the next action.
    """

    def __init__(self, *args):
        super().__init__(*args)

    @staticmethod
    def select_random_samples(
        replay_memory: List[List[Any]],
        batch_size: int,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """Select `batch_size` random samples from `replay_memory`.

        :replay_memory: list containing agent’s experiences.
        :param batch_size: number of sample to collect.
        :return: states, new_states, dones, rewards, actions.
        """
        # Select random samples
        samples = np.array(random.sample(replay_memory, batch_size))
        # Decompose each sample into arrays
        states = np.array([it[0] for it in samples]).reshape(batch_size, -1)
        new_states = np.array([it[3] for it in samples]).reshape(batch_size, -1)
        dones = np.array([it[4] for it in samples]).reshape(batch_size, -1)
        rewards = np.array([it[2] for it in samples]).reshape(batch_size, -1)
        actions = np.array([it[1] for it in samples]).reshape(batch_size, -1)

        return states, new_states, dones, rewards, actions

    @staticmethod
    def store_transition(
        replay_memory: List[List[Any]],
        state: np.array,
        action: int,
        reward: int,
        new_state: np.array,
        done: bool,
    ):
        """Store agent’s experiences in replay memory.

        :replay_memory: list containing agent’s experiences.
        :param state: agent's observation of the current environment.
        :param action: an action provided by the agent.
        :param reward: amount of reward returned after previous action.
        :param new_state: agent's observation of the next environment.
        :param done: whether the episode has ended.
        """
        replay_memory.append([state, action, reward, new_state, done])

    @staticmethod
    def update_target_weights(
        training_model: Sequential,
        target_model: Sequential,
        tau: float,
    ):
        """
        Update each target model's weights with the following rule:
                    (main_weights * tau) + (target_weights * (1 - tau))

        :param training_model: model previously trained.
        :param target_model: target model to update.
        :param tau: learning parameter.
        """
        # Get main and target models' weights
        main_weights = training_model.get_weights()
        target_weights = target_model.get_weights()
        # Update each target weight
        for i in range(len(target_weights)):
            target_weights[i] = main_weights[i] * tau + target_weights[i] * (1 - tau)
        target_model.set_weights(target_weights)

    def play_next_step(
        self,
        state: np.array,
    ) -> np.array:
        """
        Get the next action to execute given the current state: here it is only
        to initialize `get_action` method.

        :param state: agent's observation of the current environment.
        :return: next action to execute.
        """
        pass

    def play(
        self,
        n_episodes: int = 100,
        render: bool = True,
    ) -> np.array:
        """Generate `n_episodes` trials and return every scores.

        :param n_episodes: number of trials to generate (default: 100 trials).
        :param render: whether to display the environment when generating trials default: False).
        :return: list of scores of all predicted trials.
        """
        scores = []
        for episode in range(n_episodes):
            state = self.env.reset()

            score_trial = 0
            for step in range(self.n_max_steps):
                if render:
                    self.env.render()
                # Warning: `action` is related to the previous state
                # Get the next action
                action = self.play_next_step(state)
                state, reward, done, _ = self.env.step(action)
                score_trial += reward

                # Check whether the game is over or not
                if done:
                    break

            scores.append(score_trial)
            print(f"Trial {episode} / {n_episodes} - score: {score_trial}")

        print(f"Average score: {round(np.mean(scores), 1)} over {n_episodes} trials")

        return np.array(scores)


class KeyboardAgent(BaseAgent):
    """Keyboard policy. Press 'r' to restart a trial and spacebar to pause it."""

    # Up: 273 -> 65362 / Down: 274 -> 65364 / Left: 276 -> 65361 / Right: 275 -> 65363
    # Spacebar: 27 -> 32 / r: 114 -> 114
    # NanoNotes: http://en.qi-hardware.com/wiki/Key_codes
    # keysym: http://www.tcl.tk/man/tcl8.4/TkCmd/keysyms.htm
    keysyms_to_NanoNotes = {
        65362: 273,
        65364: 274,
        65363: 275,
        65361: 276,
        32: 27,
        114: 114,
    }

    def __init__(
        self,
        *args,
        keys_to_action={(275, ): 1, (276, ): 0}
    ):
        """
        :param keys_to_action: map a key to an action.
        """
        super().__init__(*args)

        self.action = 0
        self.restart = False
        self.pause = False
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
            if (key,) in self.keys_to_action:
                self.action = self.keys_to_action[(key,)]

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
            if (key,) in self.keys_to_action:
                if self.action == self.keys_to_action[(key,)]:
                    self.action = self.NO_ACTION_KEY

    def play(
        self,
        n_episodes: int = 1,
        render: bool = True
    ) -> np.array:
        """Generate `n_episodes` trials and return every scores.

        :param n_episodes: number of trials to generate (default: 1 trial).
        :param render: whether to display the environment (always true, only used to normalize `play` method)
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

                sleep(0.09)

            scores.append(score_trial)
            print(f"Trial score: {score_trial}")
            k_episode += 1

        return np.array(scores)


class RandomAgent(SmartAgent):
    """Random policy."""

    def play_next_step(
        self,
        state: np.array,
    ) -> np.array:
        """Get the next action to execute given the current state following the random policy.

        :param state: agent's observation of the current environment.
        :return: next action to execute.
        """
        return np.random.randint(0, self.action_space_size)


class LoadedAgent(SmartAgent):
    """Load a pre-trained model."""

    def __init__(self, *args):
        super().__init__(*args)
        self.model = load_model(self.model_path)
        print("Model loaded.")

    def play_next_step(
        self,
        state: np.array,
    ) -> np.array:
        """Get the next action to execute given the current state following the model's predictions.

        :param state: agent's observation of the current environment.
        :return: next action to execute.)
        """
        return np.argmax(self.model.predict(state.reshape(1, self.state_space_size)))


class NaiveLearningAgent(SmartAgent):
    """Policy learnt from best games of a random agent."""

    def __init__(self, *args):
        super().__init__(*args)

        t0 = time()
        self.model = create_random_mlp(
            n_features=self.state_space_size,
            n_categories=self.action_space_size,
            loss="categorical_crossentropy",
        )
        self.train_model()
        print(f"Model trained in {time() - t0:.2f}s.")
        if self.model_path:
            save_model(self.model, self.model_path)

    def play_next_step(
        self,
        state: np.array,
    ) -> np.array:
        """Get the next action to execute given the current state following the model's predictions.

        :param state: agent's observation of the current environment.
        :return: next action to execute.)
        """
        return np.argmax(self.model.predict(state.reshape(1, self.state_space_size)))

    def get_training_data(
        self,
    ) -> Tuple[np.array, np.array, np.array]:
        """
        Generate `n_training_episodes` with `n_max_steps` and keep only trials' data with a
        score greater than `min_score`.

        : return: state of each steps and their related actions.
        """
        # Minimum score of a trial take into account in training data
        min_score = self.reward_threshold - (0.5 * abs(self.reward_threshold))
        # Number of trials to gather training data
        n_episodes = int(10e4)

        x_train = list()  # Store every states of all trials
        y_train = list()  # Store every actions of all trials
        scores = list()  # Store every scores of all trials

        for _ in range(n_episodes):
            state = self.env.reset()

            score_trial = 0
            trial_states = list()  # Store every states of the current trial
            trial_action = list()  # Store every actions of the current trial

            for step in range(self.n_max_steps):
                # Warning: `action` is related to the previous state
                # select a random action
                action = np.random.randint(0, self.action_space_size)
                new_state, reward, done, _ = self.env.step(action)

                trial_states.append(state)
                trial_action.append(action)
                score_trial += reward
                state = new_state

                # Check whether the game is over or not
                if done:
                    break

            if score_trial > min_score:
                x_train.extend(trial_states)
                y_train.extend(trial_action)
                scores.append(score_trial)

        return (
            np.array(x_train),
            to_categorical(np.array(y_train), num_classes=self.action_space_size),
            np.array(scores),
        )

    def train_model(self):
        """Gather training data and train a basic MLP model."""
        x_train, y_train, scores = self.get_training_data()
        # Check whether training data is empty or not
        if x_train.size:
            print(f"Training score average: {round(np.mean(scores),)}")
            print(f"Training score median: {round(np.median(scores),)} \n")
            self.model.fit(x_train, y_train, epochs=5)

        else:
            print("Empty training set. Lower the minimum traing score.")


class DQNAgent(SmartAgent):
    """Policy learnt through deep Q-learning (Deep Q-Network)."""

    def __init__(self, *args):
        super().__init__(*args)

        self.replay_memory: List[List[Any]] = list()
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = 0.125
        self.batch_size = 32
        self.training_render = False
        self.n_target_success = 10
        self.max_training_episode = 500
        self.save = False

        t0 = time()
        self.model = create_small_mlp(
            n_features=self.state_space_size,
            n_categories=self.action_space_size,
            loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate),
        )
        self.target_model = create_small_mlp(
            n_features=self.state_space_size,
            n_categories=self.action_space_size,
            loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate),
        )
        self.n_training_success = 0
        self.train_model()
        print(f"Model trained in {time() - t0:.2f}s.")
        if self.model_path:
            save_model(self.model, self.model_path)

    def play_next_step(
        self,
        state: np.array,
    ) -> np.array:
        """Get the next action to execute given the current state following the model's predictions.

        :param state: agent's observation of the current environment.
        :return: next action to execute.
        """
        return np.argmax(self.model.predict(state.reshape(1, self.state_space_size)))

    def get_action(
        self,
        state: np.array,
    ) -> int:
        """
        Select a random action with probability `epsilon` or select the best
        action provided by the model.

        :param state: agent's observation of the current environment.
        :return: selected action.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])

    def replay(self):
        """Experience replay: train model from `batch_size` random previous agent’s experiences."""
        # Check there is enough sample to feed the model
        if len(self.replay_memory) < self.batch_size:
            return

        # select `batch_size` random samples from `replay_memory`
        states, new_states, dones, rewards, actions = self.select_random_samples(
            self.replay_memory, self.batch_size
        )
        # Find best actions predicted by current Q-values
        Q_values = self.target_model.predict(states)
        Q_next = np.amax(self.target_model.predict(new_states), axis=1).reshape(
            self.batch_size, -1
        )
        # Terminal states' rewards are not changed, but cumulative discounted reward is added to
        # non-terminal states' rewards, and Q-values are updated
        new_rewards = (rewards + ((1 - dones) * Q_next * self.gamma)).reshape(
            self.batch_size, -1
        )
        np.put_along_axis(
            Q_values, actions.reshape(self.batch_size, -1), new_rewards, axis=1
        )

        self.model.fit(states, Q_values, epochs=1, verbose=0)

    def train_model(self):
        """Train the deep Q-network (and save it)."""
        for trial in range(self.max_training_episode):
            t0 = time()
            score_trial = 0
            state = self.env.reset().reshape(1, self.state_space_size)

            for step in range(self.n_max_steps):
                if self.training_render:
                    self.env.render()
                # Warning: `action` is related to the previous state
                # Select a random action or the best prediction of the model
                action = self.get_action(state)
                new_state, reward, done, _ = self.env.step(action)
                new_state = new_state.reshape(1, self.state_space_size)
                self.store_transition(
                    self.replay_memory, state, action, reward, new_state, done
                )
                self.replay()
                self.update_target_weights(self.model, self.target_model, self.tau)

                state = new_state
                score_trial += reward

                if done:
                    break

            # Check whether the trial is completed or not
            if score_trial < self.reward_threshold:
                print(
                    f"Failed to complete in trial {trial} "
                    f"(score: {score_trial}, time: {time() - t0:.2f}s)"
                )
            else:
                self.n_training_success += 1
                print(
                    f"Success {self.n_training_success} / {self.n_target_success} - "
                    f"Completed in {trial+1} trials "
                    f"(score: {score_trial}, time: {time() - t0:.2f}s)"
                )

                if self.n_training_success >= self.n_target_success:
                    break
                else:
                    self.train_model()
                    break


class ACAgent(SmartAgent):
    """(Actor Critic)."""

    def __init__(self, *args):
        super().__init__(*args)

        self.replay_memory: List[List[Any]] = list()
        self.gamma = 0.95
        self.learning_rate = 0.01
        self.tau = 0.125
        self.huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.n_target_success = 10
        self.training_render = False
        self.max_training_episode = 1000
        self.save = False

        t0 = time()
        # Actor-critic model parameterisation
        self.model = create_actor_critic_mlp(
            self.state_space_size, self.action_space_size
        )

        self.n_training_success = 0
        self.train_model()
        print(f"Model trained in {time() - t0:.2f}s.")
        if self.model_path:
            save_model(self.model, self.model_path)

    def play_next_step(
        self,
        state: np.array,
    ) -> np.array:
        """Get the next action to execute given the current state following the model's predictions.

        :param state: agent's observation of the current environment.
        :return: next action to execute.)
        """
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        # Get the model predictions
        action_proba, critic_value = self.target_model(state)

        return np.argmax(action_proba)

    def get_action_value(
        self,
        state: np.array,
    ) -> Tuple[int, tf.Tensor, tf.Tensor]:
        """Predict an action (and its probability) and a critic value given a state.

        :param state: agent's observation of the current environment.
        :return: the predicted action, its log-probability and the predicted critic value.
        """
        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)
        # Get the model predictions
        action_proba, critic_value = self.model(state)
        # Sample action from current action probability distribution.
        action = np.random.choice(self.action_space_size, p=np.squeeze(action_proba))

        return action, tf.math.log(action_proba[0, action]), critic_value[0, 0]

    def replay(self):
        """Experience replay: train model from previous agent’s trial."""
        def get_target_values(
            trial_rewards: np.array
        ) -> np.array:
            """Compute the discounted cumulative sum of the previous trial rewards.

            :param trial_rewards: rewards of the previous trial.
            :return: discounted cumulative sum.
            """
            # Smallest number such that 1.0 + eps != 1.0
            eps = np.finfo(np.float32).eps.item()

            # Compute the discounted cumulative sum
            target_values = []
            discounted_sum = 0
            for reward in trial_rewards[::-1]:
                discounted_sum = reward + (self.gamma * discounted_sum)
                target_values.insert(0, discounted_sum)

            # Standardise target values
            target_values = np.array(target_values)
            target_values = (target_values - target_values.mean()) / (
                target_values.std() + eps
            )

            return target_values

        def compute_loss(
            loss: tf.keras.losses,
            log_probas: np.array,
            critic_values: np.array,
            target_values: np.array,
        ) -> float:
            """Calculating loss values to update our network.

            :param loss: keras implementation of the loss to use.
            :param log_probas: values of log-proba over the previous trial.
            :param critic_values: values of critic-values over the previous trial.
            :param target_values: values of target-values over the previous trial.

            """
            actor_losses = []
            critic_losses = []
            prediction_memory = zip(log_probas, critic_values, target_values)
            for log_proba, critic_values, target_value in prediction_memory:
                value_error = target_value - critic_values
                # Compute loss of the actor component
                actor_loss = -log_proba * value_error
                actor_losses.append(actor_loss)
                # Compute loss of the critic component
                critic_loss = loss(
                    tf.expand_dims(critic_values, 0), tf.expand_dims(target_value, 0)
                )
                critic_losses.append(critic_loss)

            return sum(actor_losses) + sum(critic_losses)

        # Get features of the previous trial in `replay_memory`
        trial_action_probas, trial_critic_values, trial_rewards = self.replay_memory
        # Compute the target values (discounted cumulative sum of rewards)
        target_values = get_target_values(trial_rewards)
        # Compute the loss
        actor_critic_loss = compute_loss(
            self.huber_loss, trial_action_probas, trial_critic_values, target_values
        )
        # Backpropagation to update our network
        grads = self.tape.gradient(actor_critic_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def train_model(self):
        """Train the actor-critic model (and save it)."""
        for trial in range(self.max_training_episode):
            t0 = time()
            trial_action_probas = []
            trial_critic_values = []
            trial_rewards = []
            score_trial = 0
            state = self.env.reset()

            with tf.GradientTape() as tape:
                self.tape = tape

                for step in range(self.n_max_steps):
                    if self.training_render:
                        self.env.render()
                    # Predict an action, its probability and a critic value given the currenrt state
                    action, action_log_proba, critic_value = self.get_action_value(state)
                    trial_critic_values.append(critic_value)
                    trial_action_probas.append(action_log_proba)
                    # Apply the sampled action in our environment
                    state, reward, done, _ = self.env.step(action)
                    trial_rewards.append(reward)
                    score_trial += reward

                    if done:
                        break

                self.replay_memory = [
                    trial_action_probas,
                    trial_critic_values,
                    trial_rewards,
                ]
                self.replay()
            self.update_target_weights(self.model, self.target_model, self.tau)

            # Check whether the trial is completed or not
            if score_trial < self.reward_threshold:
                print(
                    f"Failed to complete in trial {trial} "
                    f"(score: {score_trial}, time: {time() - t0:.2f}s)"
                )
            else:
                self.n_training_success += 1
                print(
                    f"Success {self.n_training_success} / {self.n_target_success} - "
                    f"Completed in {trial+1} trials "
                    f"(score: {score_trial}, time: {time() - t0:.2f}s)"
                )

                if self.n_training_success >= self.n_target_success:
                    break
                else:
                    self.train_model()
                    break
