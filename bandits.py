import numpy as np

class Bandit(object):

    def __init__(self, arm_probabilities):
        arm_probabilities = np.array(arm_probabilities).flatten()
        self.arm_probabilities = arm_probabilities / np.sum(arm_probabilities)
        self.n_arms = len(self.arm_probabilities)
        
    def optimal_action(self):
        return np.argmax(self.arm_probabilities)

    def step(self, action):
        obesrvation = None
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

class UCB1Agent(object):

    def __init__(self, n_actions):
        self.n_actions = n_actions
        self.memory = []

    def handle(self, action, observation, reward, done, info):
        self.memory.append({'action': action, 'reward': reward})
        return np.argmax(self.upper_confidence_bounds(self.memory, self.n_actions))

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

        
if __name__ == '__main__':
    easy_bandit = Bandit([0.1, 0.9])
    medium_bandit = Bandit([0.25, 0.75])
    hard_bandit = Bandit([0.4, 0.6])
    agent = UCBAgent(easy_bandit.n_actions())
    episode_length = 100

    optimal_action = agent.optimal_action
    action = None
    observation = None
    reward = 0
    done = False
    info = {}
    for i_episode in range(episode_length):
        action = agent.handle(action, observation, reward, done, info)
        observation, reward, done, info = easy_bandit.step(action)

        results.append({
            'optimal_action': agent.optimal_action()
            })
