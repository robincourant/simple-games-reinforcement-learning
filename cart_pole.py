from bokeh.plotting import show
import gym
import numpy as np

from basic_plots import frequency_plot
from agents import NaiveLearningAgent, RandomAgent, DeepQLearningAgent


def display_score_vs_threshold(
    env: gym.Env,
    n_episodes: int,
):
    """
    Display (in html format) the chart of the distribution of scores
    for a given number of trials.

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

    distribution_plot = frequency_plot(np.array(scores), max_score + 1, (0, max_score),
                                       f"Distribution of scores (for {n_episodes:.0e} trials)",
                                       "Score", "Number of trials")

    show(distribution_plot)


def main():
    # Cart-Pole-v0
    env_v0 = gym.make("CartPole-v0")
    agent = DeepQLearningAgent(env_v0, 195)
    env_v0.close()


if __name__ == '__main__':
    main()
