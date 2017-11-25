"""
Code is heavily based on this: https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/
Also based on my dqn code (especially for the nn architecture)
"""

import numpy as np
import tensorflow as tf

import gym, time, random, threading, os, pickle

from skimage.transform import resize
from skimage.color import rgb2gray

from collections import deque

import matplotlib.pyplot as plt

# constants
ENV = 'TimePilot-v0'
SAVE_DIR = ENV + '-ckpts-a3c'
SAVE_PARAM_PATH = SAVE_DIR + '/params'

NUM_ACTIONS = None # we'll change this later

RUN_TIME = 100000
THREADS = 16
OPTIMIZERS = 1
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 20
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 1
EPS_STOP = 0.1
EPS_STOP_LIST = [0.1, 0.01, 0.5]
EPS_STOP_DIST = [0.4, 0.3, 0.3]
EPS_STEPS = 400000

BATCH_SIZE = 32
TRAIN_SIZE = 10000 # updates ocassionally
FRAME_SKIP = 1
LEARN_RATE = 7e-4
LEARN_STEPS = 20000000
RMS_DECAY = 0.99

LOSS_V = 0.5 # v loss coefficeint
LOSS_ENTROPY = 0.01 # entropy coefficient

class Brain:
    def __init__(self):
        self.train_queue = []
        self.lock_queue = threading.Lock()

        self.graph = tf.Graph()
        self.graph.as_default()
        with self.graph.as_default():
            self.build_graph()
            self.init_graph()

    def init_graph(self):
        """
        Initializes graph, loads from checkpoint if necessary.
        """

        self.session = tf.Session(graph = self.graph)
        self.saver = tf.train.Saver()

        # if save_dir exists, load it
        # we want to restore from latest version
        if os.path.isdir(SAVE_DIR):
            print("Loading from file")
            ckpt = tf.train.latest_checkpoint(SAVE_DIR)
            self.num = int(ckpt[len(SAVE_PARAM_PATH) + 1:]) # get number of latest checkpoint
            self.saver.restore(self.session, ckpt)
            print("NUM: ", self.num)

        else: # otherwise initialize randomly
            print("Initializing randomly")
            self.session.run(tf.global_variables_initializer())
            self.num = 0

    def save_graph(self):
        """
        Saves graph to checkpoint.
        """

        self.saver.save(self.session, SAVE_PARAM_PATH, global_step = self.num)

    def build_graph(self):
        """
        Defines the deep network on which we train.
        Copied from the dqn implementation.
        """
        # input is 84 x 84 x 4, as per the dqn paper
        # the first index represents batch size
        self.s_t = tf.placeholder(tf.float32, shape = [None, 84, 84, 4])
        self.a_t = tf.placeholder(tf.float32, shape = [None, NUM_ACTIONS])
        self.r_t = tf.placeholder(tf.float32, shape = [None, 1])
        self.learn_rate = tf.placeholder(tf.float32, shape = [])

        # input tensor shape: [batch_size, 84, 84, 4]
        # output tensor shape: [batch_size, 21, 21, 16]
        conv1 = tf.layers.conv2d(
                inputs = self.s_t,
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
        self.out_actions = tf.layers.dense(
                inputs = dense, 
                units = NUM_ACTIONS, 
                activation = tf.nn.softmax)

        # input tensor shape: [batch_size, 256]
        # output tensor shape: [batch_size, 1]
        self.out_value = tf.layers.dense(
                inputs = dense, 
                units = 1)

        # idk this probably works
        log_prob = tf.log( tf.reduce_sum( self.out_actions * self.a_t, axis = 1, keep_dims = True) + 1e-10)
        advantage = self.r_t - self.out_value

        loss_policy = - log_prob * tf.stop_gradient(advantage)
        loss_value = LOSS_V * tf.square(advantage)
        entropy = LOSS_ENTROPY * tf.reduce_sum( self.out_actions * tf.log(self.out_actions + 1e-10), axis = 1, keep_dims = True)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(self.learn_rate, decay = RMS_DECAY)

        grads, tvars = zip(*optimizer.compute_gradients(loss_total))
        clipped_grads, _ = tf.clip_by_global_norm(grads, 40)
        self.minimize = optimizer.apply_gradients(zip(clipped_grads, tvars))

        '''
        #WIP
        optimizer = tf.train.AdamOptimizer(self.learn_rate)
        grads, tvars = zip(*optimizer.compute_gradients(loss_total))

        lens = [4096, 16, 8192, 32, 991232, 256, 1536, 6, 256, 1]
        grads = [tf.reshape(grads[i], [lens[i]]) for i in range(10)]
        tvars = [tf.reshape(tvars[i], [lens[i]]) for i in range(10)]
        print(grads[0].shape)
        print(tvars[0].shape)
        clipped_grads, _ = tf.clip_by_global_norm(grads, 40)
        print(clipped_grads[0].shape)
        self.minimize = optimizer.apply_gradients(zip(clipped_grads, tvars))
        '''

    def get_learn_rate(self):
        if self.num > LEARN_STEPS: return 0
        return LEARN_RATE * (1 - self.num / LEARN_STEPS)

    def optimize(self):
        if len(self.train_queue) < TRAIN_SIZE:
            time.sleep(0) # yield
            return

        with self.lock_queue:

            # I don't really understand this tbh
            if len(self.train_queue) < TRAIN_SIZE:
                return

            train_queue = self.train_queue
            self.train_queue = []

        print("Start training network...{}".format(self.num))
        random.shuffle(train_queue) # decrease correlation
        print("LENGTH: ", len(train_queue))
        s, a, r, s_, done = zip(*train_queue)
        s = np.array(s)
        a = np.array(a)
        r = np.reshape(np.array(r), (-1, 1))
        s_ = np.array(s_)
        done = np.reshape(np.array(done), (-1, 1))

        n = len(s)
        for i in range(0, n, BATCH_SIZE):
            end = min(n, i + BATCH_SIZE)
            print("Training {} to {}".format(i, end))

            v = self.predict_v(s_[i:end])
            r_ = r[i:end] + GAMMA_N * v * done[i:end]

            self.session.run(self.minimize, feed_dict = {self.s_t: s[i:end], self.a_t: a[i:end], self.r_t: r_, self.learn_rate: self.get_learn_rate()})

        self.save_graph()

        print("Finished training network!")

    def train_push(self, s, a, r, s_, done):
        # don't push in too many samples
        # this does mean that there are inefficiencies here
        if len(self.train_queue) >= TRAIN_SIZE: return

        with self.lock_queue:
            self.train_queue.append((s, a, r, s_, int(done)))
            self.num += 1
            #print(self.num)

    def predict(self, s):
        return self.session.run([self.out_actions, self.out_value], feed_dict = {self.s_t: s})

    def predict_p(self, s):
        return self.session.run(self.out_actions, feed_dict = {self.s_t: s})

    def predict_v(self, s):
        return self.session.run(self.out_value, feed_dict = {self.s_t: s})

class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = [] # for n-step return
        self.R = 0.
        self.frames = 0

    def get_eps(self):
        if self.frames >= self.eps_steps:
            return self.eps_end
        else:
            return self.eps_start + self.frames * (self.eps_end - self.eps_start) / self.eps_steps

    def act(self, s):
        eps = self.get_eps()

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            p = brain.predict_p(np.array([s]))[0]

            #a = np.argmax(p) # hard decision
            a = np.random.choice(NUM_ACTIONS, p = p) # soft decision

            return a
    
    def train(self, s, a, r, s_, done):
        # I haven't gone through the code yet, just assumed it works
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_onehot = np.zeros(NUM_ACTIONS)
        a_onehot[a] = 1

        self.memory.append( (s, a_onehot, r, s_) )

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if done:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_, done)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0
        
        if len(self.memory) >= N_STEP_RETURN:
            n = N_STEP_RETURN
            s, a, r, s_ = get_sample(self.memory, n)
            brain.train_push(s, a, r, s_, done)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

class Environment(threading.Thread):
    def __init__(self, render = False, eps_start = EPS_START, eps_end = EPS_STOP, eps_steps = EPS_STEPS):
        threading.Thread.__init__(self)

        self.stop_signal = False
        self.render = render
        self.env = gym.make(ENV)
        self.agent = Agent(eps_start, eps_end, eps_steps)
        self.phi = np.zeros([84, 84, 4])
        self.s_prev = np.zeros([210, 160, 3])

        self.R_list = [] # for plotting

    def process_image(self, s, s_prev):
        """
        Processes image by converting from [210, 160, 3] -> [84, 84]
        skimage just does all this lol
        thanks random github guy
        """

        s = np.maximum(s, s_prev)
        return resize(rgb2gray(s), (110, 84))[13:97, :]
    
    def update_phi(self, s):
        """
        Updates phi with new observation.
        Since phi only stores the last 4 observations, 
        we rollback phi and insert s (like a queue).
        """

        s_new = np.reshape(self.process_image(s, self.s_prev), (84, 84, 1))
        self.phi = np.concatenate([self.phi[:, :, 1:], s_new], axis = 2)
        self.s_prev = s

    def run_episode(self):
        self.phi = np.zeros([84, 84, 4])
        s = self.env.reset()
        self.update_phi(s)

        R = 0
        while True:
            time.sleep(THREAD_DELAY) # yield

            a = self.agent.act(self.phi)

            for i in range(FRAME_SKIP): # perform a for FRAME_SKIP times
                s_, r, done, info = self.env.step(a)
                phi_bef = self.phi
                self.update_phi(s_)

                if self.render:
                    self.env.render()
                    #pass
                else:
                    self.agent.train(phi_bef, a, r, self.phi, done)

                R += r

                if done or self.stop_signal:
                    break
            
            if done or self.stop_signal:
                break

        print("Total R:", R)
        if self.render:
            self.R_list.append(R)

    def run(self):
        while not self.stop_signal:
            self.run_episode()

    def stop(self):
        self.stop_signal = True


class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True

def main(unused_argv):
    env_test = Environment(render = True, eps_start = 0, eps_end = 0)

    global NUM_ACTIONS
    NUM_ACTIONS = env_test.env.action_space.n
    global brain
    brain = Brain() 

    envs = [Environment(eps_end = np.random.choice(EPS_STOP_LIST, p = EPS_STOP_DIST)) for i in range(THREADS)]
    opts = [Optimizer() for i in range(OPTIMIZERS)]

    #we run all of these guys in parallel
    for o in opts: o.start()
    for e in envs: e.start()
    #env_test.start()

    time.sleep(RUN_TIME)

    for e in envs: e.stop()
    for e in envs: e.join()

    for o in opts: o.stop()
    for o in opts: o.join()
    #env_test.stop()
    #env_test.join()

    print("Training finished")
    #env_test.run()

    '''
    with open('R_data', 'wb') as f:
        pickle.dump(env_test.R_list, f)
    '''

    '''
    plt.plot(env_test.R_list)
    plt.ylabel("Episodic reward")
    plt.xlabel("Episode number")
    plt.title("Performance on Pong")
    plt.show()
    '''



if __name__ == '__main__':
    tf.app.run()

