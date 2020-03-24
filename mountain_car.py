import gym

from agents import DeepQLearningAgent


def main():
    env = gym.make("MountainCar-v0")
    agent = DeepQLearningAgent(env, -190)
    env.close()


if __name__ == '__main__':
    main()
