import numpy as np

class UCB1Agent(object):

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.memory = []

    def act(self, observation=None):
        return np.argmax(self.upper_confidence_bounds(self.memory, self.n_actions))

    def handle_transition(self, observation=None, action=None, reward=0, next_observation=None, done=False):
        self.memory.append({'action': action, 'reward': reward})

    def reset(self):
        self.memory = []

    def upper_confidence_bounds(self, memory, n_actions):
        # For UCB1:
        # https://www.cs.bham.ac.uk/internal/courses/robotics/lectures/ucb1.pdf
        # For many more UCB choices:
        # http://banditalgs.com/2016/09/18/the-upper-confidence-bound-algorithm/ 
        estimates = np.ones(n_actions) * np.inf
        for action in range(n_actions):
            examples = filter(lambda e: e['action'] == action, memory)
            rewards = [e['reward'] for e in examples]
            if len(rewards) > 0:
                mean_reward = sum(rewards) / len(rewards)
                confidence_interval = np.sqrt( (2*np.log(len(memory))) / len(rewards))
                ucb_action = mean_reward + confidence_interval
                estimates[action] = ucb_action

        return estimates
