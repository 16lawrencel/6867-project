# everybody in tensorflow does this
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym

import random
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

tf.logging.set_verbosity(tf.logging.INFO)

class Agent:
    def __init__(self):
        self.env = gym.make('CartPole-v0') # environment
        self.num_actions = self.env.action_space.n # number of possible actions
        self.capacity = 1000000 # replay memory capacity
        self.train_time = 10000000 # number of total frames to train
        self.D = deque([], maxlen = self.capacity) # replay memory
        self.eps = 1 # eps-greedy
        self.k = 4 # frame skipping
        self.done = True # is it the start of a new episode?
        self.phi = np.zeros([1, 84, 84, 4]) # processed inputs
        self.gamma = 0.9
        self.batch_size = 32 # batch size for experience replay

        self.graph = tf.Graph()
        self.build_graph()

    def build_graph(self):
        """
        Defines the deep q-network on which we train.
        """
        # input is 84 x 84 x 4, as per the dqn paper
        # the first index represents batch size
        self.input_layer = tf.placeholder(tf.float32, shape = [None, 84, 84, 4])
        batch_size = self.input_layer.shape[0]

        # y_ and a_ are the labels and actions specified in 
        # the training data. They're used to compute loss.
        self.y_ = tf.placeholder(tf.float32, shape = [None])
        self.a_ = tf.placeholder(tf.int32, shape = [None])

        # input tensor shape: [batch_size, 84, 84, 4]
        # output tensor shape: [batch_size, 21, 21, 16]
        conv1 = tf.layers.conv2d(
                inputs = self.input_layer, 
                filters = 16, 
                kernel_size = [8, 8], 
                strides = 4, 
                padding = 'same', 
                activation = tf.nn.relu)
        assert(conv1.shape[1:4] == [21, 21, 16])

        # input tensor shape: [batch_size, 21, 21, 16]
        # output tensor shape: [batch_size, 11, 11, 32]
        conv2 = tf.layers.conv2d(
                inputs = conv1, 
                filters = 32, 
                kernel_size = [4, 4], 
                strides = 2, 
                padding = 'same', 
                activation = tf.nn.relu)
        assert(conv2.shape[1:4] == [11, 11, 32])

        # flatten tensor into a batch of vectors
        # input tensor shape: [batch_size, 11, 11, 32]
        # output tensor shape: [batch_size, 11 * 11 * 32]
        conv2_flat = tf.reshape(conv2, [-1, 11 * 11 * 32])

        # input tensor shape: [batch_size, 11 * 11 * 32]
        # output tensor shape: [batch_size, 256]
        dense = tf.layers.dense(
                inputs = conv2_flat, 
                units = 256, 
                activation = tf.nn.relu)
        print(dense.shape)
        assert(dense.shape[1] == 256)

        # input tensor shape: [batch_size, 256]
        # output tensor shape: [batch_size, num_actions]
        # output is estimated Q values
        Q_vals = tf.layers.dense(
                inputs = dense, 
                units = self.num_actions)
    
        # one hot representation of data set actions
        # input size: [batch_size]
        # output size: [batch_size, num_actions]
        a_one_hot = tf.one_hot(self.a_, depth = self.num_actions,
                on_value = True, off_value = False, dtype = tf.bool)

        # then we just select the actions we want from Q_vals
        Q_vals_a = tf.reshape(tf.boolean_mask(Q_vals, a_one_hot), [-1])
        self.loss = tf.reduce_sum((self.y_ - Q_vals_a) ** 2)

        self.optimizer = tf.train.RMSPropOptimizer(0.1).minimize(self.loss)

        self.pred = tf.argmax(Q_vals, axis = 1)
        self.Q_max = tf.reduce_max(Q_vals, axis = 1)

    def step_eps(self):
        """
        Steps eps. Recall that eps decreases linearly from 1 to 0.1 
        in the first 10^6 steps, then stays constant at 0.1.
        """
        self.eps = max(0.1, self.eps - 9e-7)

    def process_image(self, obs):
        """
        Processes image by converting from [210, 160, 3] -> [84, 84]
        skimage just does all this lol
        thanks random github guy
        """
        return resize(rgb2gray(obs), (110, 84))[13:97, :]

    def update_phi(self, obs):
        """
        Updates phi with new observation.
        Since phi only stores the last 4 observations, 
        we rollback phi and insert obs (like a queue).
        """
        obs = np.reshape(self.process_image(obs), (1, 110, 84, 1))
        self.phi = np.concatenate(self.phi[:, :, :, 1:], obs)

    def train(self, session):
        """
        Performs experience replay.
        Trains using a random minibatch of size batch_size and 
        using RMSProp.
        """

        batch = random.sample(self.D, self.batch_size)

        batch_input = np.concatenate(list(map(lambda x : x[3], batch)), axis = 0)
        Q_max = session.run(self.Q_max, feed_dict = {self.input_layer: batch_input})
        Q_max = np.array(Q_max)

        done_mask = np.array(list(map(lambda x : int(x[4]), batch)))
        y_ = np.array(list(map(lambda x : x[2], batch))) + self.gamma * Q_max * done_mask
        a_ = np.array(list(map(lambda x : x[1], batch)))
        batch_input = np.concatenate(list(map(lambda x : x[0], batch)), axis = 0)
        # train gradient descent
        session.run(self.optimizer, feed_dict = {self.input_layer: batch_input, 
            self.y_: y_, self.a_: a_})

    def step_env(self, session, t):
        """
        Performs k steps in the environment, then records the image 
        on the kth step and trains with experience replay.
        """

        if t % 1000 == 0:
            print('Step {}:'.format(t))

        # start a new episode
        if self.done:
            obs = self.env.reset()
            self.update_phi(obs) # I don't think we need this actually
            self.done = False

        if np.random.rand() <= self.eps: # choose random action
            action = self.env.action_space.sample()
        else: # choose best action, according to estimated Q-values
            action = np.asscalar(session.run(self.pred, feed_dict = {self.input_layer: self.phi}))

        # step action for k frames
        for i in range(self.k):
            obs, reward, self.done, info = self.env.step(action)
            if self.done: break

        # clip reward
        reward = 1 if reward > 0 else -1 if reward < 0 else 0

        phi_bef = self.phi
        self.update_phi(obs)

        # add experience tuple for experience replay
        self.D.append((phi_bef, action, reward, self.phi, self.done))

        # experience replay
        self.train(session)

        self.step_eps()

    def run_env(self):
        """
        Runs and trains in the environment for M * k frames.
        Basically just calls step_env M times.
        """
        with tf.Session(graph = self.graph) as session:
            session.run(tf.global_variables_initializer())
            print('Initializing variables')
            for t in range(self.train_time):
                self.step_env(session, t)

def main(unused_argv):
    agent = Agent()
    agent.run_env()

if __name__ == '__main__':
    tf.app.run()

