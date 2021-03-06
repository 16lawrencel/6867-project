# everybody in tensorflow does this
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import gym

import _pickle as pickle
import os
import sys
import time
import random
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque

tf.logging.set_verbosity(tf.logging.INFO)

class Agent:
    def __init__(self):
        self.env_name = 'Pong-v0' # environment name
        self.env = gym.make(self.env_name) # environment
        self.num_actions = self.env.action_space.n # number of possible actions
        self.capacity = 10000 # replay memory capacity
        self.train_time = 100000 # number of total frames to train
        self.D = deque([], maxlen = self.capacity) # replay memory
        self.eps = 1 # eps-greedy
        self.eps_step = 0.9 / (self.train_time / 10)
        self.k = 4 # frame skipping
        self.done = True # is it the start of a new episode?
        self.phi = np.zeros([1, 84, 84, 4]) # processed inputs
        self.gamma = 0.9
        self.batch_size = 32 # batch size for experience replay
        self.save_dir = self.env_name + '-ckpts' # save directory for checkpoints
        self.save_param_path = self.save_dir + '/params' # save file for parameters
        self.save_dq_path = self.save_dir + '/deque.p' # save file for replay memory deque

        self.graph = tf.Graph()
        with self.graph.as_default():
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
        self.eps = max(0.1, self.eps - self.eps_step)

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
        obs = np.reshape(self.process_image(obs), (1, 84, 84, 1))
        self.phi = np.concatenate([self.phi[:, :, :, 1:], obs], axis = 3)

    def train(self, session):
        """
        Performs experience replay.
        Trains using a random minibatch of size batch_size and 
        using RMSProp.
        """

        batch = random.sample(self.D, min(self.batch_size, len(self.D)))

        batch_input = np.concatenate(list(map(lambda x : x[3], batch)), axis = 0)
        Q_max = session.run(self.Q_max, feed_dict = {self.input_layer: batch_input})

        done_mask = np.array(list(map(lambda x : int(x[4]), batch)))
        y_ = np.array(list(map(lambda x : x[2], batch))) + self.gamma * Q_max * done_mask
        a_ = np.array(list(map(lambda x : x[1], batch)))

        batch_input = np.concatenate(list(map(lambda x : x[0], batch)), axis = 0)

        # train gradient descent
        session.run(self.optimizer, feed_dict = {self.input_layer: batch_input, self.y_: y_, self.a_: a_})

    def step_env(self, session, t):
        """
        Performs k steps in the environment, then records the image 
        on the kth step and trains with experience replay.
        """

        self.env.render() # comment this out if you don't want to see awesome video game playing :'(

        print('Step {}'.format(t))

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

    def dump_deque(self):
        """
        Dumps deque from saved deque file.
        Since the deque gets very large, we use a hack to allow pickle to work (by breaking it into chunks of size 2**31 - 1)
        See https://stackoverflow.com/questions/42653386/does-pickle-randomly-fail-with-oserror-on-large-files
        """

        max_bytes = 2**31 - 1
        bytes_out = pickle.dumps(self.D)
        n_bytes = sys.getsizeof(bytes_out)
        with open(self.save_dq_path, 'wb') as f_out:
            for i in range(0, n_bytes, max_bytes):
                f_out.write(bytes_out[i:i + max_bytes])

    def load_deque(self):
        """
        Load deque from saved deque file.
        Since the deque gets very large, we use a hack to allow pickle to work (by breaking it into chunks of size 2**31 - 1)
        See https://stackoverflow.com/questions/42653386/does-pickle-randomly-fail-with-oserror-on-large-files
        """

        max_bytes = 2**31 - 1
        input_size = os.path.getsize(self.save_dq_path)
        bytes_in = bytearray(0)
        with open(self.save_dq_path, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)

        self.D = pickle.loads(bytes_in)

    def run_env(self):
        """
        Runs and trains in the environment for M * k frames.
        Basically just calls step_env M times.
        """

        with tf.Session(graph = self.graph) as session:
            saver = tf.train.Saver() # to load and save checkpoints
            # if save_dir exists, load it
            # we want to restore from latest version
            if os.path.isdir(self.save_dir):
                print("Loading from file")
                ckpt = tf.train.latest_checkpoint(self.save_dir)
                num = int(ckpt[len(self.save_param_path) + 1:]) # get number of latest checkpoint
                print('num: ', num)
                saver.restore(session, ckpt)
                start_t = num
                self.eps = max(0.1, 1 - num * self.eps_step)

                # now we load the deque of experience replay
                self.load_deque()
                
            else: # otherwise initialize randomly
                print("Initializing randomly")
                session.run(tf.global_variables_initializer())
                start_t = 0

            for t in range(start_t + 1, self.train_time):
                self.step_env(session, t)
                # we save checkpoint every 10000 steps
                if t % 10000 == 0:
                    print("Saving {}...".format(t))
                    saver.save(session, self.save_param_path, global_step = t)
                    self.dump_deque()

def main(unused_argv):
    agent = Agent()
    agent.run_env()

if __name__ == '__main__':
    tf.app.run()

