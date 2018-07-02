import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import time

# Hyper Parameters for DQN
GAMMA = 0.9  # discount factor for target Q
INITIAL_EPSILON =0.5  # starting value of epsilon
FINAL_EPSILON = 0.01  # final value of epsilon
REPLAY_SIZE = 10000  # experience replay buffer size
BATCH_SIZE = 32  # size of minibatch


class DQN():
    # DQN Agent
    def __init__(self, env):
        # init experience replay
        self.replay_buffer = deque()
        # init some parameters
        self.time_step = 0
        self.epsilon = INITIAL_EPSILON
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n


        self.create_Q_network()
        self.create_training_method()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # network weights
        # build one hidden layer neural network with 56 nodes
        W1 = self.weight_variable([self.state_dim, 56])
        b1 = self.bias_variable([56])
        # build one hidden layer neural network with 128 nodes
        W2 = self.weight_variable([56,128])
        b2 = self.bias_variable([128])
        #
        W3 = self.weight_variable([128, self.action_dim])
        b3 = self.bias_variable([self.action_dim])
        # input layer
        self.state_input = tf.placeholder("float", [None, self.state_dim])
        # hidden layers
        # activate function : ReLU
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        h2_layer = tf.nn.relu(tf.matmul(h_layer, W2) + b2)
        # Q Value layer
        # i.e.output layer
        self.Q_value = tf.matmul(h2_layer, W3) + b3

    def create_training_method(self):
        # define placeholder for Q-network input and output

        # input (batch_size * action_dim)
        self.action_input = tf.placeholder("float", [None, self.action_dim])  # one hot presentation
        self.y_input = tf.placeholder("float", [None])
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        # learning rate and optimizer
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1
        # Step 1: obtain random minibatch from replay memory
        # element of minibatch (state, one_hot_action, reward, next_state, done)
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        # state
        state_batch = [data[0] for data in minibatch]
        # action
        action_batch = [data[1] for data in minibatch]
        # reward
        reward_batch = [data[2] for data in minibatch]
        # next state
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate y
        y_batch = []
        # calculate Q_value: feed state_input to Deep Q-network to predict Q_value
        # use the parameter in current step to predict next_state reward 
        # and then use gradient to update network parameter
        # find the best action for -next- state batch
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})
        # according to paper, calculate reward
        for i in range(0, BATCH_SIZE):
            # done is the flag to show whether the game is finished
            done = minibatch[i][4]
            # if terminated reward does not change
            # else : add the max reward in next state
            # we create the labeled data for training as a regression problem
            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)

        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000

    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)


# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 10000  # Episode limitation
STEP = 500  # Step limitation in an episode
TEST = 10  # The number of experiment test every 100 episode
EXCEPT_REWARD = 300


def main(is_training):
    # initialize OpenAI Gym env and dqn agent
    CURRENT_REWARD = 150
    if is_training:
        env = gym.make(ENV_NAME)
        agent = DQN(env)
        env._max_episode_steps = EXCEPT_REWARD
        for episode in range(EPISODE):
            # initialize task
            state = env.reset()
            # Train
            for step in range(STEP):
                action = agent.egreedy_action(state)  # e-greedy action for train
                next_state, reward, done, _ = env.step(action)
                # Define reward for agent
                reward_agent = -1 if done else 0.1
                agent.perceive(state, action, reward, next_state, done)
                state = next_state
                if done:
                    break
            # Test every 100 episodes
            if episode % 100 == 0:
                total_reward = 0
                for i in range(TEST):
                    state = env.reset()
                    for j in range(STEP):
                        # env.render()
                        action = agent.action(state)  # direct action for test
                        state, reward, done, _ = env.step(action)
                        total_reward += reward
                        if done:
                            print('Finished at step ',j)
                            break
                ave_reward = total_reward / TEST
                print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
                if ave_reward >= CURRENT_REWARD:
                    CURRENT_REWARD = ave_reward
                    print('Start to save model...')
                    saver = tf.train.Saver(max_to_keep=1)
                    model_path = "model1/model.ckpt"
                    save_path = saver.save(agent.session, model_path)
                    print('Finished!')
                    if ave_reward >= EXCEPT_REWARD-1:
                        break

    else:
        env = gym.make(ENV_NAME)
        env._max_episode_steps = 1000
        agent = DQN(env)
        saver = tf.train.Saver()
        saver.restore(agent.session, tf.train.latest_checkpoint("model1/"))
        state = env.reset()
        total_reward = 0
        while True:
            env.render()
            action = agent.action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
            time.sleep(1/12)
            if done:
                print('Game Over!, total_reward is ',total_reward)
                break

if __name__ == '__main__':
    main(is_training = False)
