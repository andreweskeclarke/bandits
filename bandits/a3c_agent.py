import collections
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
        self.state_h = None

    def act(self, observation=None):
        action_probs, _, self.state_h = self.brain.single_prediction(observation, self.state_h)
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions-1)
        else:
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
        n_inputs = self.brain.n_inputs
        default_o = np.zeros((n_inputs,))
        observations = np.zeros((len(memory), n_inputs))
        actions = np.zeros((len(memory), self.n_actions))
        rewards = np.zeros((len(memory), 1))
        next_observations = np.zeros((len(memory), n_inputs))
        discounts = np.zeros((len(memory), 1))
        end_of_episode_mask = np.ones((len(memory), 1))
        end_of_episode_mask[-1][0] = 0.0
        for i in range(len(memory)):
            o, a, r, o_ = memory[i]
            o = o if o is not None else default_o
            o_ = o_ if o_ is not None else default_o
            r = 0.0
            for j in range(i, len(memory)):
                r += memory[j][2] * (self.gamma**(j-i))

            v_discount = self.gamma**(len(memory) - i)

            observations[i][:] = o.reshape((n_inputs,))
            actions[i][:] = a
            rewards[i][0] = r
            next_observations[i][:] = o_.reshape((n_inputs,))
            discounts[i][0] = v_discount
        brain.push_training_episode(**{
                'observation': np.array(observations),
                'action': actions,
                'reward': rewards,
                'next_observation': np.array(next_observations),
                'discount': discounts,
                'mask': end_of_episode_mask,
                })


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

