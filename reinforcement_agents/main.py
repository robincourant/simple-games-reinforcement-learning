import argparse

from bokeh.io import show
import gym
import numpy as np

from reinforcement_agents.agents import (
    ACAgent,
    DQNAgent,
    KeyboardAgent,
    NaiveLearningAgent,
    RandomAgent,
)
from utils.basic_plots import frequency_plot


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
}


def play_game():
    """Play a given game with a given agent in command line."""
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
    args = parser.parse_args()

    env = gym.make(AVAILABLE_GAMES[args.game])
    agent = AVAILABLE_AGENTS[args.agent](env, args.threshold)
    agent.play(n_episodes=args.n_episodes, render=args.display)
    env.close()


def display_random_score_vs_threshold(
    env: gym.Env,
    n_episodes: int,
):
    """
    Display (in html format) the chart of the distribution of scores
    of a random agent for a given number of trials.

    :param env: gym enviroment object.
    :param n_episodes: number of trials to generate.
    """
    max_score = env.spec.max_episode_steps
    random_agent = RandomAgent(env=env)
    scores = random_agent.play(n_episodes=n_episodes)

    samples_distribution = list()
    for _threshold in np.arange(0, max_score, 5):
        inds, = np.where(scores > _threshold)
        samples_distribution.append(inds.size)

    distribution_plot = frequency_plot(
        np.array(scores), max_score + 1, (0, max_score),
        f"Distribution of scores (for {n_episodes:.0e} trials)",
        "Score", "Number of trials")

    show(distribution_plot)


if __name__ == "__main__":
    play_game()
