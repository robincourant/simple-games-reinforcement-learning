import gym

from agents import NaiveLearningAgent


def main():
    env = gym.make("CartPole-v0")
    agent = NaiveLearningAgent(
        env=env,
        min_score=50,
        n_training_episodes=10000,
        n_training_steps=500,
        n_test_episodes=100,
        n_test_steps=500,
        training_render=False,
        test_render=False,
    )
    agent.predict()
    env.close()


if __name__ == '__main__':
    main()
