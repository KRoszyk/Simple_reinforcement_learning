import gym
import time
import numpy as np
import random
import os


def ex_1():
    env = gym.make('CartPole-v1')
    env.seed(42)

    # print(env.action_space)
    # print(env.observation_space)              # showing values of env
    # print(env.observation_space.high)
    # print(env.observation_space.low)

    action = 1

    env.reset()
    for _ in range(1000):
        env.render()
        time.sleep(0.1)
        # action = env.action_space.sample()    # random choice of action
        if action:  # simple try
            action = 0
        else:
            action = 1
        print(action)
        observation, reward, done, info = env.step(action)
        print(f'observation: {observation}, reward: {reward}')
        if done:
            print("the end!")
            time.sleep(1)
    env.close()


def ex_2():
    env = gym.make('Taxi-v3')
    env.seed(42)

    # print(env.action_space)
    # print(env.observation_space)  # showing values of env

    q_table = np.zeros((500, 6), dtype=np.float32)
    lr = 0.1
    discount_factor = 0.6
    no_training_episodes = 10000
    epsilon = 0.5

    for i in range(no_training_episodes):  # training code
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            if random.uniform(0, 1) < epsilon:
                # exploration
                action = env.action_space.sample()  # random choice of action
            else:
                # exploitation
                action = np.argmax(q_table[observation])

            next_observation, reward, done, info = env.step(action)
            total_reward += reward
            # print(f'observation: {observation}, reward: {reward}, info: {info}')

            max_next_observation = np.max(q_table[next_observation])

            q_table[observation, action] = (1 - lr) * q_table[observation, action] + lr * (
                        reward + discount_factor * max_next_observation)

            observation = next_observation

        if i % 100 == 0:
            print(f'total_reward episode {i + 1}: {total_reward}')

    no_test_episodes = 100

    for i in range(no_test_episodes):  # testing code
        observation = env.reset()
        env.render()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(q_table[observation])
            observation, reward, done, info = env.step(action)
            total_reward += reward
            print(f'observation: {observation}, reward: {reward}, info: {info}')
            os.system('cls')
            env.render()
            time.sleep(0.1)
            print(f'total_reward episode {i + 1}: {total_reward}')

    env.close()


def ex_3():
    env = gym.make('FrozenLake-v0')
    env.seed(42)

    print(env.action_space)
    print(env.observation_space)  # showing values of env

    q_table = np.zeros((env.observation_space.n, env.action_space.n), dtype=np.float32)
    lr = 0.1
    discount_factor = 0.6
    no_training_episodes = 10000
    epsilon = 0.5

    print("Training time!")
    for i in range(no_training_episodes):  # training code
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            if random.uniform(0, 1) < epsilon:
                # exploration
                action = env.action_space.sample()  # random choice of action
            else:
                # exploitation
                action = np.argmax(q_table[observation])

            next_observation, reward, done, info = env.step(action)
            total_reward += reward
            # print(f'observation: {observation}, reward: {reward}, info: {info}')

            max_next_observation = np.max(q_table[next_observation])

            q_table[observation, action] = (1 - lr) * q_table[observation, action] + lr * (
                        reward + discount_factor * max_next_observation)

            observation = next_observation

        if i % 100 == 0:
            print(f'total_reward episode {i + 1}: {total_reward}')

    no_test_episodes = 100

    print("Testing time!")
    for i in range(no_test_episodes):  # testing code
        observation = env.reset()
        env.render()
        done = False
        total_reward = 0
        while not done:
            action = np.argmax(q_table[observation])
            observation, reward, done, info = env.step(action)
            total_reward += reward
            print(f'observation: {observation}, reward: {reward}, info: {info}')
            os.system('cls')
            env.render()
            time.sleep(0.1)
            print(f'total_reward episode {i + 1}: {total_reward}')

    env.close()


if __name__ == '__main__':
    ex_2()
