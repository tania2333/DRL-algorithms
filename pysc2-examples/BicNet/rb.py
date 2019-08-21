"""
Data structure for implementing experience replay

Author: Patrick Emami
"""
from collections import deque
import random
import numpy as np
import tensorflow as tf

class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123, num_agents=9, vector_obs_len=33, output_len=3, sess):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)
        self.state_t = tf.placeholder(tf.float32, shape=(None, num_agents, vector_obs_len))
        self.action_t = tf.placeholder(tf.float32, shape=(None, num_agents, output_len))
        self.reward_t = tf.placeholder(tf.float32, shape=(None, num_agents, 1))
        self.state2_t = tf.placeholder(tf.float32, shape=(None, num_agents, vector_obs_len))
        self.done_t = tf.placeholder(tf.bool, shape=(None, num_agents, 1))
        self.sess = sess

    def add(self, state, action, reward, done, state2):
        self.sess.run(feed_dict={
            self.state_t[next]: state,
            self.action_t[next]: action
        })
        experience = (state, action, reward, done, state2)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        np.save('buffer_null.npy', self.buffer)

    def size(self):
        return self.count

    def sample_batch_9agents(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([vectorConcate(_[0]) for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([vectorConcate(_[4]) for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

def vectorConcate(screen):
    screen_final = screen
    for i in range(8):
      screen_final = np.concatenate((screen_final,screen),axis=0)
    return  screen_final