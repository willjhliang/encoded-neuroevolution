import time
import gym
env = gym.make('SpaceInvaders-v0')

for i in range(1):
    state = env.reset()
    totalReward = 0

    for _ in range(1000):
        env.render()

        # take a random action
        randomAction = env.action_space.sample()
        observation, reward, done, info = env.step(randomAction)

        time.sleep(0.1)
        totalReward += reward

    print('Episode', i, 'Total reward:', totalReward)

env.close()

