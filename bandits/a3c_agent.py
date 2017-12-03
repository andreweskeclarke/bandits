import random
import threading
import numpy as np


# Inspired from:
# https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
# and the corresponding article at:
# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/

class A3CAgent(object):

    def __init__(self, n_actions, brain, epsilon=0.0, gamma=0.99):
        self.n_actions = n_actions
        self.brain = brain
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = []

    def act(self, observation=None):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        else:
            action_probs = self.brain.predict_action(observation)
            return np.random.choice(self.n_actions, p=action_probs)

    def handle_transition(self, observation=None, action=None, reward=0, next_observation=None):
        one_hot_actions = np.zeros(self.n_actions)
        one_hot_actions[action] = 1
        self.memory.append((observation, one_hot_actions, reward, next_observation))
        if observation is not None and next_observation is None:
            self.push_to_brain(self.brain, self.memory)

    def reset(self):
        self.memory = list()

    def push_to_brain(self, brain, memory):
        for i in range(len(memory)):
            o, a, r, o_ = memory[i]
            if o is None:
                o = self._none_state()
            if o_ is None:
                o_ = self._none_state()
            r = 0.0
            for j in range(i, len(memory)):
                r += memory[j][2] * (self.gamma**(j-i))

            brain.push_training_example(
                    observation=o,
                    action=a,
                    reward=r,
                    next_observation=o_)

    def _none_state(self):
        return np.zeros(self.n_actions)


class A3CEnvironment(object):

    def __init__(self, stop_signal, agent, env):
        self.stop_signal = stop_signal
        self.agent = agent
        self.env = env

    def run(self):
        while not self.stop_signal.is_set():
            self.run_one_episode()

    def run_one_episode(self):
        pass

class Optimizer(threading.Thread):
    def __init__(self, brain, stop_signal):
        threading.Thread.__init__(self)
        self.brain = brain
        self.stop_signal = stop_signal

    def run(self):
        while not self.stop_signal.is_set():
            self.brain.optimize()

