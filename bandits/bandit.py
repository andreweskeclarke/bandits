import numpy as np
import random
import math

class Bandit(object):

    def __init__(self, arm_probabilities, episode_length=np.inf, include_steps=False, shuffle_probs=True):
        self._parse_arm_probabilities(arm_probabilities)
        self._episode_length = episode_length
        self._steps = 0
        self._include_steps = include_steps
        self._shuffle_probs = shuffle_probs

    def _parse_arm_probabilities(self, arm_probabilities):
        arm_probabilities = np.array(arm_probabilities).flatten()
        self.arm_probabilities = arm_probabilities / np.sum(arm_probabilities)
        self.n_arms = len(self.arm_probabilities)

    def optimal_action(self):
        return np.argmax(self.arm_probabilities)

    def step(self, action):
        if self._steps >= self._episode_length:
            raise ValueError('This environment has been exhausted, please reset()')

        reward = np.random.binomial(1, self.arm_probabilities[int(action)])
        self._steps += 1
        if self._steps < self._episode_length:
            observation = np.zeros((self.n_inputs(),))
            observation[int(action)] = 1.0
            if self._include_steps:
                observation[-2] = reward
                observation[-1] = self._steps
            else:
                observation[-1] = reward
            done = False
        else:
            observation = None
            done = True

        info = {}
        return observation, reward, done, info

    def reset(self):
        if self._shuffle_probs:
            self._parse_arm_probabilities(
                    random.sample(self.arm_probabilities.tolist(), len(self.arm_probabilities.tolist())))
        self._steps = 0
        return None

    def render(self):
        pass

    def n_inputs(self):
        # Input and Outputs include one-hot encoding of actions, the reward, and potentially the step number
        if self._include_steps:
            return self.n_actions() + 2
        else:
            return self.n_actions() + 1

    def n_actions(self):
        return self.n_arms


class MultiBandit(object):

    def __init__(self, envs, episode_length, include_steps=False):
        self.envs = envs
        self._episode_length = episode_length
        self._steps = 0
        self._include_steps = include_steps
        if self._include_steps:
            for e in envs:
                e._include_steps = True

    def _get_env_for_step(self, step, envs, episode_length):
        step = max(0, min(episode_length-1, step))
        percent_done = float(step) / float(episode_length)
        current_env_index = math.floor(percent_done * float(len(envs)))
        current_env_index = max(0, min(len(envs)-1, current_env_index))
        return envs[current_env_index]

    def episode_length(self):
        return self._episode_length

    def optimal_action(self):
        return self._get_env_for_step(self._steps, self.envs, self.episode_length()).optimal_action()

    def step(self, action):
        if self._steps >= self.episode_length():
            raise ValueError('This environment has been exhausted, please reset()')

        _, reward, done, info = self._get_env_for_step(self._steps, self.envs, self.episode_length()).step(action)
        self._steps += 1
        if self._steps < self.episode_length():
            observation = np.zeros((self.n_inputs(),))
            observation[int(action)] = 1.0
            if self._include_steps:
                observation[-2] = reward
                observation[-1] = self._steps
            else:
                observation[-1] = reward
            done = False
        else:
            observation = None
            done = True
        return observation, reward, done, info

    def reset(self):
        self._steps = 0
        for e in self.envs:
            e.reset()
        return None

    def render(self):
        self._get_env_for_step(self._steps, self.envs, self.episode_length()).render()

    def n_inputs(self):
        return self._get_env_for_step(self._steps, self.envs, self.episode_length()).n_inputs()

    def n_actions(self):
        return self._get_env_for_step(self._steps, self.envs, self.episode_length()).n_actions()


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

