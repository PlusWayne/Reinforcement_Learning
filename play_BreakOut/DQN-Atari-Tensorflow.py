import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
import time

CROPPED_IMAGE_SIZE = 84
ACTION_DIM = 4
Learning_rate = 0.0001
MAX_EPISODE = 10000
IMAGE_SEQUENCE_SIZE = 4
OBSERVE = 2000
BATCH_SIZE = 32
GAMMA = 0.9
MEMORY_SIZE = 10000

def preprocessing(img):
    gray_scaled_image = tf.image.rgb_to_grayscale(img)
    resized_image = tf.image.resize_images(gray_scaled_image, [110, 84])
    cropped_image = tf.image.resize_image_with_crop_or_pad(resized_image, CROPPED_IMAGE_SIZE, CROPPED_IMAGE_SIZE)
    return cropped_image


def Create_Q_network():
    W_conv1 = tf.Variable(tf.truncated_normal([8, 8, 4, 16]))
    b_conv1 = tf.Variable(tf.constant(0.01, shape=[16]))

    W_conv2 = tf.Variable(tf.truncated_normal([4, 4, 16, 32], stddev=0.01))
    b_conv2 = tf.Variable(tf.constant(0.01, shape=[32]))

    W_fc1 = tf.Variable(tf.truncated_normal([2592, 256]))
    b_fc1 = tf.Variable(tf.constant(0.01, shape=[256]))

    W_fc2 = tf.Variable(tf.truncated_normal([256, ACTION_DIM]))
    b_fc2 = tf.Variable(tf.constant(0.01, shape=[ACTION_DIM]))

    return W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2


def forward_pass(image_input, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2):
    image_input = tf.reshape(image_input, shape=[-1, 84, 84, 4])

    h_conv1 = tf.nn.relu(tf.nn.conv2d(image_input, W_conv1, [1, 4, 4, 1], padding='VALID') + b_conv1)
    # h_conv1 [None, 20, 20, 16] ceil(84-8+1/4)
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, [1, 2, 2, 1], padding="VALID") + b_conv2)
    # h_conv2 [None, 9, 9 ,32] ceil(20-4+1/2)
    h_conv2_flat = tf.reshape(h_conv2, [-1, 2592])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
    q_value = tf.matmul(h_fc1, W_fc2) + b_fc2

    return q_value

def get_action_matrix(val):
    actions = np.zeros(ACTION_DIM)
    actions[val] = 1
    return actions



def main():
    observation_image = tf.placeholder(tf.float32, shape=(210, 160, 3))
    processed_image = preprocessing(observation_image)
    image_input = tf.placeholder(tf.float32, shape=(None, 4, 84, 84, 1))
    W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2 = Create_Q_network()
    q = forward_pass(image_input, W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2)
    action_index = tf.argmax(q, 1)

    # train
    action_input = tf.placeholder(tf.float32, [None, ACTION_DIM])
    y_input = tf.placeholder(tf.float32, [None])
    Q_action = tf.reduce_sum(tf.multiply(q, action_input), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y_input - Q_action))
    train = tf.train.AdamOptimizer(Learning_rate).minimize(cost)

    memory = deque()
    init_op = tf.global_variables_initializer()

    saver = tf.train.Saver()

    env = gym.make('Breakout-v0')
    image_sequence = []

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)
        t = 0
        ckpt = tf.train.get_checkpoint_state('./tmp/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, './tmp/model.ckpt')
            print('Successfully loaded network weights')
        else:
            print('Could not find old network weights')

        for episode in range(0, MAX_EPISODE):
            observation = env.reset()

            reward_sum = 0

            while True:
                # env.render()
                p_image = sess.run(processed_image, feed_dict={observation_image: observation})
                image_sequence.append(p_image)

                # default action
                action = 0

                if len(image_sequence) <= IMAGE_SEQUENCE_SIZE:
                    next_observation, reward, done, info = env.step(action)
                else:
                    image_sequence.pop(0)
                    current_state = np.stack(
                        [image_sequence[0], image_sequence[1], image_sequence[2], image_sequence[3]])

                    # get action
                    e = 0.1
                    if np.random.rand(1) < e:
                        actioin = env.action_space.sample()
                    else:
                        action, _q = sess.run([action_index, q], feed_dict={image_input: [current_state]})

                        next_observation, reward, done, info = env.step(action) # take a random action

                        # Store in experience relay
                        p_image = sess.run(processed_image, feed_dict={observation_image: next_observation})
                        next_state = np.stack([image_sequence[1], image_sequence[2], image_sequence[3], p_image])
                        
                        #print(get_action_matrix(action))
                        action_state = get_action_matrix(action)
                        memory.append((current_state, action_state, reward, next_state, done)) 

                        if len(memory) > MEMORY_SIZE:
                            memory.pop(0)

                # training
                if t > OBSERVE:
                    # step 1: get batch
                    minibatch = random.sample(memory,BATCH_SIZE)
                    state_batch = [data[0] for data in minibatch]
                    action_batch = [data[1] for data in minibatch]
                    reward_batch = [data[2] for data in minibatch]
                    nextState_batch = [data[3] for data in minibatch]
                    terminal_batch = [data[4] for data in minibatch]

                    # step 2: calculate y
                    y_batch = []
                    QValue_batch = sess.run(q, feed_dict={image_input: nextState_batch})

                    for i in range(0,BATCH_SIZE):
                        terminal = minibatch[i][4]
                        if terminal:
                            y_batch.append(reward_batch[i])
                        else:
                            y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))
                    sess.run(train, feed_dict={y_input: y_batch, action_input: action_batch, image_input: state_batch})

                reward_sum += reward
                t += 1

                observation = next_observation
                if t % 1000 == 0:
                    saver.save(sess, './tmp/model.ckpt')
                if done:
                    episode += 1
                    print('Episode {} Rewards: {}'.format(episode, reward_sum))
                    break;

if __name__ == '__main__':
    main()