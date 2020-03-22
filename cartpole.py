from bokeh.plotting import show
import gym
import numpy as np

from basic_plots import frequency_plot
from agents import NaiveLearningAgent, RandomAgent


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
    random_agent = RandomAgent(
        env=env,
        n_episodes=n_episodes,
        render=False,
    )
    scores = random_agent.play()

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
    agent = NaiveLearningAgent(
        env=env_v0,
        min_score=70,
        n_training_episodes=int(10e3),
        n_testing_episodes=100,
        training_render=False,
        testing_render=False,
    )
    scores_v0 = agent.play()
    print(f"Average score for 100 trials (v0): {np.mean(scores_v0)}")
    print()
    env_v0.close()

    # Cart-Pole-v1
    env_v1 = gym.make("CartPole-v1")
    # display_score_vs_threshold(env_v1, int(10e3))
    env_v1.close()

    # agent = NaiveLearningAgent(
    #     env=env_v1s,
    #     min_score=50,
    #     n_training_episodes=10000,
    #     n_testing_episodes=100,
    #     training_render=False,
    #     testing_render=False,
    # )
    # scores_v1 = agent.play()
    # print(f"Average score for 100 trials (v1): {np.mean(scores_v1)}")
    # print()


if __name__ == '__main__':
    main()
