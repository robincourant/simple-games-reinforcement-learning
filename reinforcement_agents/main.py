import argparse

import gym
import numpy as np

from reinforcement_agents.agents import (
    ACAgent,
    DQNAgent,
    KeyboardAgent,
    LoadedAgent,
    NaiveLearningAgent,
    RandomAgent,
)

AVAILABLE_GAMES = {
    "cart-pole": "CartPole-v0",
    "mountain-car": "MountainCar-v0",
}
AVAILABLE_AGENTS = {
    "keyboard": KeyboardAgent,
    "random": RandomAgent,
    "naive": NaiveLearningAgent,
    "deep-q-network": DQNAgent,
    "actor-critic": ACAgent,
    "loaded": LoadedAgent,
}


def play_game():
    """Play a given game with a given agent in command line."""
    # TODO: default keyboard, loaded path and save model
    parser = argparse.ArgumentParser("Play a given game with a given agent")

    parser.add_argument(
        "game", choices=AVAILABLE_GAMES.keys(),
        help="choose a game in the list"
    )
    parser.add_argument(
        "agent", choices=AVAILABLE_AGENTS.keys(),
        help="choose an agent in the list"
    )
    parser.add_argument(
        "-n", "--n-episodes",
        default=10, type=int, metavar='',
        help="number of episodes to play"
    )
    parser.add_argument(
        "-t", "--threshold",
        default=-np.inf, type=int, metavar='',
        help="score threshold to win an episode (unset by default)"
    )
    parser.add_argument(
        "-d", "--display",
        action="store_true",
        help="whether to display or not the environment"
    )
    parser.add_argument(
        "-p", "--model-path",
        help="path to save the trained model. If not provided, it will not be saved."
    )
    args = parser.parse_args()

    env = gym.make(AVAILABLE_GAMES[args.game])
    agent = AVAILABLE_AGENTS[args.agent](env, args.threshold, args.model_path)
    agent.play(n_episodes=args.n_episodes, render=args.display)
    env.close()


if __name__ == "__main__":
    play_game()
