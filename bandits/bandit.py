import numpy as np
import random

class Bandit(object):

    def __init__(self, arm_probabilities):
        arm_probabilities = np.array(arm_probabilities).flatten()
        self.arm_probabilities = arm_probabilities / np.sum(arm_probabilities)
        self.n_arms = len(self.arm_probabilities)

    def optimal_action(self):
        return np.argmax(self.arm_probabilities)

    def step(self, action):
        observation = None
        reward = np.random.binomial(1, self.arm_probabilities[action])
        done = False
        info = {}
        return observation, reward, done, info

    def reset(self):
        pass

    def render(self):
        pass

    def n_actions(self):
        return self.n_arms

def fixed_bandit():
    return Bandit(random.sample([0.0, 1.0], 2))

def easy_bandit():
    return Bandit(random.sample([0.1, 0.9], 2))

def medium_bandit():
    return Bandit(random.sample([0.25, 0.75], 2))

def hard_bandit():
    return Bandit(random.sample([0.4, 0.6], 2))

def random_bandit():
    return Bandit(random.sample([0.5, 0.5], 2))

