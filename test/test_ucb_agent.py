import unittest
from bandits.ucb_agent import *

class TestUCBAgent(unittest.TestCase):

    def test_ucb_default_actions(self):
        n_actions = 2
        agent = UCB1Agent(n_actions)
        data = [{'action':0, 'reward':1}]
        estimates = agent.upper_confidence_bounds(data, n_actions)
        self.assertEqual(1, np.argmax(estimates))

        n_actions = 5
        agent = UCB1Agent(n_actions)
        data = [ 
                {'action':0, 'reward':1},
                {'action':1, 'reward':1},
                {'action':2, 'reward':1},
                ]
        estimates = agent.upper_confidence_bounds(data, n_actions)
        self.assertEqual(3, np.argmax(estimates))

    def test_ucb_not_ordering_dependent(self):
        n_actions = 2
        agent = UCB1Agent(n_actions)
        data = [ 
                {'action':0, 'reward':1},
                {'action':1, 'reward':0},
                {'action':0, 'reward':2},
                {'action':1, 'reward':2},
                ]
        estimates_sequence_1 = agent.upper_confidence_bounds(data, n_actions)
        self.assertEqual(0, np.argmax(estimates_sequence_1))

        data = [ 
                {'action':1, 'reward':2},
                {'action':1, 'reward':0},
                {'action':0, 'reward':1},
                {'action':0, 'reward':2},
                ]
        estimates_sequence_2 = agent.upper_confidence_bounds(data, n_actions)
        self.assertEqual(0, np.argmax(estimates_sequence_2))

        self.assertTrue(np.equal(estimates_sequence_1, estimates_sequence_2).all())

    def test_ucb_calculations(self):
        n_actions = 2
        agent = UCB1Agent(n_actions)
        data = [ ]
        estimates = agent.upper_confidence_bounds(data, n_actions)
        expected = np.array([np.inf, np.inf])
        self.assertTrue(np.equal(estimates, expected).all())

        data = [ 
                {'action':0, 'reward':1},
                ]
        estimates = agent.upper_confidence_bounds(data, n_actions)
        expected = np.array([1.0, np.inf])
        self.assertTrue(np.isclose(estimates, expected).all())

        data = [ 
                {'action':0, 'reward':1},
                {'action':1, 'reward':1},
                ]
        estimates = agent.upper_confidence_bounds(data, n_actions)
        expected = np.array([2.17741002, 2.17741002])
        self.assertTrue(np.isclose(estimates, expected).all())

        data = [ 
                {'action':0, 'reward':1},
                {'action':0, 'reward':1},
                ]
        estimates = agent.upper_confidence_bounds(data, n_actions)
        expected = np.array([1.83255461, np.inf])
        self.assertTrue(np.isclose(estimates, expected).all())


    def test_three_armed_bandit_where_first_arm_always_succeeds(self):
        n_actions = 3
        agent = UCB1Agent(n_actions)
        actions = [None,0,1,2,0,0,0,0,1,2,0,0,0,0,0,0,0,0,1,2,0,0]
        rewards = [None,1,0,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1]
        for t in range(len(actions) - 1):
            self.assertEqual(actions[t+1], agent.handle(action=actions[t], observation=None, reward=rewards[t], done=False, info={}))
    def test_three_armed_bandit_where_middle_arm_always_succeeds(self):
        n_actions = 3
        agent = UCB1Agent(n_actions)
        actions = [None,0,1,2,1,1,1,1,0,2,1,1,1,1,1,1,1,1,0,2,1,1]
        rewards = [None,0,1,0,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1]
        for t in range(len(actions) - 1):
            self.assertEqual(actions[t+1], agent.handle(action=actions[t], observation=None, reward=rewards[t], done=False, info={}))

    def test_three_armed_bandit_where_last_arm_always_succeeds(self):
        n_actions = 3
        agent = UCB1Agent(n_actions)
        actions = [None,0,1,2,2,2,2,2,0,1,2,2,2,2,2,2,2,2,0,1,2,2]
        rewards = [None,0,0,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,1,1]
        for t in range(len(actions) - 1):
            self.assertEqual(actions[t+1], agent.handle(action=actions[t], observation=None, reward=rewards[t], done=False, info={}))


if __name__ == '__main__':
    unittest.main()
