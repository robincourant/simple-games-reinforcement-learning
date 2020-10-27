# Get Started

Set a python 3.6 virtual environment and set your `PYTHONPATH` env variable to the root of the project is it not the case.

Then run the following command to install all necessary libraries:
```
pip install -r requirements.txt
```

To launch a 15 games of cart-pole with a score threshold of 195.0 and a deep-Q-network agent, use the following command:
```
python reinforcement_agents/main.py -n 15 -t 195 cart-pole deep-q-network
```

For more details run:
```
python reinforcement_agents/main.py -h
```

Models should be saved in a folder `simple-games-reinforcement-learning/saved_models/`, it will be ignored by git.

# Games

There are currently 2 different games:
  - Cart-pole: this game is solved if the average score over 100 consecutive trials is greater than or equal to 195.
  - Mountain-car: this game is solved if the average score over 100 consecutive trials is greater than or equal to -110.


# Agents

## Type of agents

There are currently 5 different type of agents able to play:
  - `KeyboardAgent`: this agent is controlled with the keyboard. Press 'r' to restart a trial and spacebar to pause it, otherwise, playing keys depend on the game.
  - `RandomAgent`: this agent chooses its next move randomly.
  - `NaiveLearningAgent`: this agent generates a training set with random moves, and learns from the best ones.
  - `DQNAgent`: or Deep-Q-Network agent approximates the policy with a neural network.
  - `ACAgent`: or Actor-Critic agent is a deep-Q-network agent (actor network) with an extra network (critic network) that evaluates and influences the actor network decision.

The `LoadedAgent` allows to load a pre-trained agent/model.

