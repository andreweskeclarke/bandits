import random
import threading
import numpy as np
import time

# Inspired from:
# https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py
# and the corresponding article at:
# https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/

class A3CAgent(object):

    def __init__(self, n_actions, brain, epsilon=0.0, gamma=0.99, thread_delay=0.001):
        self.n_actions = n_actions
        self.brain = brain
        self.epsilon = epsilon
        self.gamma = gamma
        self.memory = []
        self._thread_delay = thread_delay

    def act(self, observation=None):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        else:
            action_probs = self.brain.predict_action(observation)
            if action_probs.ndim > 1:
                action_probs  = action_probs[0]
            return np.random.choice(self.n_actions, p=action_probs)

    def handle_transition(self, observation=None, action=None, reward=0, next_observation=None, done=False):
        one_hot_actions = np.zeros(self.n_actions)
        one_hot_actions[action] = 1
        self.memory.append((observation, one_hot_actions, reward, next_observation))
        if done:
            self.push_to_brain(self.brain, self.memory)

    def reset(self):
        self.memory = list()

    def push_to_brain(self, brain, memory):
        for i in range(len(memory)):
            o, a, r, o_ = memory[i]
            r = 0.0
            for j in range(i, len(memory)):
                r += memory[j][2] * (self.gamma**(j-i))

            time.sleep(self._thread_delay) # yield to allow many many parallel agents running
            brain.push_training_example(
                    observation=o,
                    action=a,
                    reward=r,
                    next_observation=o_)


class AsynchRunner(threading.Thread):

    def __init__(self, stop_signal, agent, env, thread_delay=0.001):
        threading.Thread.__init__(self)
        self.stop_signal = stop_signal
        self.agent = agent
        self.env = env
        self._thread_delay = thread_delay
        self._total_episodes = 0

    def run(self):
        while not self.stop_signal.is_set():
            self.run_one_episode(agent=self.agent, env=self.env)

    def run_one_episode(self, agent, env):
        agent.reset()
        action = None
        observation = None
        next_observation = env.reset()
        reward = 0
        done = False
        info = {}
        while not done and not self.stop_signal.is_set():
            self._total_episodes += 1
            time.sleep(self._thread_delay) # yield to allow many many parallel agents running

            observation = next_observation
            action = agent.act(observation=observation)
            next_observation, reward, done, info = env.step(action=action)
            agent.handle_transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation,
                    done=done)


class AsynchOptimizer(threading.Thread):

    def __init__(self, brain, stop_signal):
        threading.Thread.__init__(self)
        self.brain = brain
        self.stop_signal = stop_signal

    def run(self):
        while not self.stop_signal.is_set():
            time.sleep(0)
            self.brain.optimize()

