import tensorflow as tf
import numpy as np
import random
import cv2
import sys

sys.path.append('game')
import wrapped_flappy_bird as fb

ACTIONS = 2
GAMMA = 0.99
OBSERVE = 10000
EXPLORE = 3000000
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH = 32
IMAGE_SIZE = 80

S = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 4), name='State')
A = tf.placeholder(tf.float32, shape=(None, ACTIONS), name='Action')

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
T_3 = 
