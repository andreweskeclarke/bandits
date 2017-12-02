import numpy as np
import random

class Bandit(object):

    def __init__(self, arm_probabilities):
        self._parse_arm_probabilities(arm_probabilities)

    def _parse_arm_probabilities(self, arm_probabilities):
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
        self._parse_arm_probabilities(
                random.sample(self.arm_probabilities.tolist(), len(self.arm_probabilities.tolist())))

    def render(self):
        pass

    def n_actions(self):
        return self.n_arms

def fixed_bandit():
    probs = [0.0, 1.0]
    return Bandit(probs)

def easy_bandit():
    probs = [0.1, 0.9]
    return Bandit(probs)

def medium_bandit():
    probs = [0.25, 0.75]
    return Bandit(probs)

def hard_bandit():
    probs = [0.4, 0.6]
    return Bandit(probs)

def random_bandit():
    probs = [0.5, 0.5]
    return Bandit(probs)

