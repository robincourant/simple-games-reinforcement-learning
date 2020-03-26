import gym

from agents import (
    DeepQLearningAgent,
    # KeyboardAgent,
    # NaiveLearningAgent,
    # RandomAgent,
)


def main():
    reward_threshold = -190
    env = gym.make("MountainCar-v0")
    agent = DeepQLearningAgent(env, reward_threshold)
    agent.play(n_episodes=100, render=True)
    env.close()


if __name__ == '__main__':
    main()
