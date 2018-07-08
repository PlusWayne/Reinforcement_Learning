import gym

game_name = 'Breakout-v0'
env = gym.make(game_name)
print(env.action_space.n)
# for i_episode in range(1):
#     observation = env.reset()
#     for t in range(100):
#         env.render()
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             break