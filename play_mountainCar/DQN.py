import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import time

EPSILON = 1
Learning_rate = 0.001
REPLAY_SIZE = 100000
BATCH_SIZE = 64
GAMMA = 0.99


# create DQN class
class DQN():
    # init
    def __init__(self, env, train_writer=None):
        self.replay_buffer = deque()
        self.time_step = 0
        self.epsilon = EPSILON
        self.state_dim = env.observation_space.shape[0]

        self.action_dim = env.action_space.n
        self.train_writer = train_writer

        self.create_Q_network()
        self.create_training_method()

        # init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())

    def create_Q_network(self):
        # network weights
        # build one hidden layer neural network with 64 nodes
        W1 = tf.Variable(tf.truncated_normal([self.state_dim, 64]))
        b1 = tf.Variable(tf.constant(value=0.01, shape=[64]))
        # build one hidden layer neural network with 128 nodes
        # W2 = tf.Variable(tf.truncated_normal([64, 128]))
        # b2 = tf.Variable(tf.constant(value=0.01, shape=[128]))
        #
        W2 = tf.Variable(tf.truncated_normal([64, self.action_dim]))
        b2 = tf.Variable(tf.constant(value=0.01, shape=[self.action_dim]))
        # input layer
        self.state_input = tf.placeholder(tf.float32, [None, self.state_dim])
        # hidden layers
        # activate function : ReLU
        h_layer = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
        # h2_layer = tf.nn.relu(tf.matmul(h_layer, W2) + b2)
        # Q Value layer
        # i.e.output layer
        self.Q_value = tf.matmul(h_layer, W2) + b2
        #self.writer = tf.summary.FileWriter('F:/Reinforcement_Learning/play_mountainCar/log',tf.get_default_graph())
        #self.writer.close()

    def create_training_method(self):
        # define placeholder for Q-network

        # one hot representation
        self.action_input = tf.placeholder(tf.float32, [None, self.action_dim])

        # 'true' value
        self.y_input = tf.placeholder(tf.float32, [None])

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), axis=1)

        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))

        #tf.summary.scalar('Loss',self.cost)
        #self.merged = tf.summary.merge_all()

        self.optimizer = tf.train.AdamOptimizer(Learning_rate).minimize(self.cost)

    def perceive(self, state, action, reward, next_state, done):
        # define buffer
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1

        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

        if len(self.replay_buffer) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        self.time_step += 1

        # 1 get batch from replay_buffer
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # 2 calculate y
        y_batch = []
        Q_value_batch = self.Q_value.eval(feed_dict={self.state_input: next_state_batch})

        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]

            if done:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        #summary = self.session.run(self.merged,feed_dict={
            # self.state_input: state_batch,
            # self.action_input: action_batch,
            # self.y_input: y_batch})
        #self.train_writer.add_summary(summary,global_step=self.time_step)

        self.optimizer.run(feed_dict={
            self.state_input: state_batch,
            self.action_input: action_batch,
            self.y_input: y_batch
        })

    def egreedy_action(self, state):
        Q_value = self.Q_value.eval(feed_dict={self.state_input: [state]})[0]

        if random.random() <= self.epsilon:
        	random_action = random.randint(0, self.action_dim - 1)
        	if random_action == 1:
        		random_action = 2
        	return random_action
        else:
            return np.argmax(Q_value)

    def action(self, state):
        action = np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])
        return action if action !=1 else 2 


EPISODE = 10000  # Episode limitation
STEP = 4000  # Step limitation in an episode
TEST = 5  # The number of experiment test every 100 episode
ENV_NAME = 'MountainCar-v0' # 'CartPole-v0' 'MountainCar-v0'

def main(is_training):
    if is_training:
        CURRENT_REWARD = -150
        env = gym.make(ENV_NAME) 
        env._max_episode_steps = 4000
        agent = DQN(env=env, train_writer = tf.summary.FileWriter('F:/Reinforcement_Learning/play_mountainCar/log'))

        for episode in range(EPISODE):
            state = env.reset()
            for step in range(STEP):
                action = agent.egreedy_action(state)
                next_state, reward, done, _ = env.step(action)
                # reward = 1 if done else 0
                # perceive and train
                agent.perceive(state, action, reward, next_state, done)
                state = next_state
                if done:
                    print('episode:',episode ,'step:', step)
                    agent.epsilon *= 0.95
                    agent.epsilon = max(0.1,agent.epsilon)
                    break
                    # Test every 100 episodes
            if episode % 100 == 0 and episode != 0:
                total_reward = 0
                for i in range(TEST):
                    state = env.reset()
                    for j in range(STEP):
                        # env.render()
                        action = agent.action(state)  # direct action for test
                        state, reward, done, _ = env.step(action)
                        total_reward += reward
                        if done:
                            break
                ave_reward = total_reward / TEST
                print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)
                if ave_reward >= CURRENT_REWARD:
                    CURRENT_REWARD = ave_reward
                    print('Start to save model...')
                    saver = tf.train.Saver(max_to_keep=1)
                    model_path = "model/model.ckpt"
                    save_path = saver.save(agent.session, model_path)
                    print('Finished!')
                if ave_reward >= -150:
                    break
    else:
        env = gym.make(ENV_NAME)
        env._max_episode_steps = 1000
        agent = DQN(env)
        saver = tf.train.Saver()
        saver.restore(agent.session, tf.train.latest_checkpoint("model/"))
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
