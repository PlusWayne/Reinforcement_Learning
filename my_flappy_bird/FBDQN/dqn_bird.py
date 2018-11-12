import tensorflow as tf
import numpy as np
import random
import cv2
import sys

sys.path.append('game')
import wrapped_flappy_bird as fb
from collections import deque

ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000
EXPLORE = 3000000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
IMAGE_SIZE = 80

# define Q-network
S = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 4), name='State')
A = tf.placeholder(tf.float32, shape=(None, ACTIONS), name='Action')
Y = tf.placeholder(tf.float32, shape=(None), name='Y')

T_1 = tf.layers.conv2d(S, kernel_size=8, filters=32, padding='same',
                       kernel_initializer=tf.truncated_normal_initializer(0, 0.01),
                       bias_initializer=tf.constant_initializer(0.01))
T_1 = tf.nn.relu(T_1)
T_1 = tf.layers.max_pooling2d(T_1, pool_size=2, strides=2, padding='same')

T_2 = tf.layers.conv2d(T_1, kernel_size=4, filters=64, padding='same',
                       kernel_initializer=tf.truncated_normal_initializer(0, 0.01),
                       bias_initializer=tf.constant_initializer(0.01))
T_2 = tf.nn.relu(T_2)

T_3 = tf.layers.conv2d(T_2, kernel_size=3, filters=64, padding='same',
                       kernel_initializer=tf.truncated_normal_initializer(0, 0.01),
                       bias_initializer=tf.constant_initializer(0.01))
T_3 = tf.nn.relu(T_3)

T_4 = tf.contrib.layers.flatten(T_3)

T_4 = tf.layers.dense(T_4, units=512, activation=tf.nn.relu, bias_initializer=tf.constant_initializer(0.01))

Q = tf.layers.dense(T_4, units=ACTIONS, bias_initializer=tf.constant_initializer(0.01), name='Q-value')
Q_ = tf.reduce_sum(tf.multiply(Q, A), axis=1)
loss = tf.losses.mean_squared_error(Y, Q_)
optimizer = tf.train.AdamOptimizer(1e-6).minimize(loss)

# game

game_state = fb.GameState()
reply = deque()

do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
img, reward, terminal = game_state.frame_step(do_nothing)
img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
S0 = np.stack((img, img, img, img), axis=2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

t = 0
success = 0
saver = tf.train.Saver(max_to_keep=1)
epsilon = INITIAL_EPSILON

while True:
    if epsilon > FINAL_EPSILON and t > OBSERVE:
        epsilon = INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE * (t - OBSERVE)

    Q_value = Q.eval(feed_dict={S: [S0]}, session=sess)[0]  # the output is [[value]]. So use [0] to remove.
    action = np.zeros(ACTIONS)
    if np.random.random() <= epsilon:
        action_index = np.random.randint(ACTIONS)
    else:
        action_index = np.argmax(Q_value)
    action[action_index] = 1

    img, reward, terminal = game_state.frame_step(action)
    if reward == 1:
        success += 1

    img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = np.reshape(img, (IMAGE_SIZE, IMAGE_SIZE, 1))
    S1 = np.append(S0[:, :, 1:], img, axis=2)

    reply.append((S0, action, reward, S1, terminal))

    if len(reply) > REPLAY_MEMORY:
        reply.popleft()

    if t > OBSERVE:
        minibatch = random.sample(reply, BATCH)
        S_batch = [tt[0] for tt in minibatch]
        A_batch = [tt[1] for tt in minibatch]
        R_batch = [tt[2] for tt in minibatch]
        S1_batch = [tt[3] for tt in minibatch]
        T_batch = [tt[4] for tt in minibatch]

        Y_batch = []
        Q1_batch = Q.eval(feed_dict={S: S1_batch}, session=sess)
        for i in range(BATCH):
            if T_batch[i]:
                Y_batch.append(R_batch[i])
            else:
                Y_batch.append(R_batch[i] + GAMMA * np.max(S1_batch[i]))
        sess.run(optimizer, feed_dict={S: S_batch, A: A_batch, Y: Y_batch})

    S0 = S1
    t += 1
    if t % 200==0:
        print(t)
    if t % 1000 == 0 and t > OBSERVE:
        print(
            'the loss is {} at {} step'.format(loss.eval(feed_dict={S: S_batch, A: A_batch, Y: Y_batch}, session=sess),
                                               t))

    if t > OBSERVE and t % 100000 == 0:
        saver.save(sess, './flappy_bird_dqn', global_step=t)
