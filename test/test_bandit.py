import unittest
from bandits.bandit import *

class TestBandit(unittest.TestCase):

    def test_optimal_action(self):
        self.assertEqual(0, Bandit([1.0, 1.0, 1.0]).optimal_action())
        self.assertEqual(0, Bandit([1.0, 0.0, 0.0]).optimal_action())
        self.assertEqual(1, Bandit([0.0, 2.0, 0.0]).optimal_action())
        self.assertEqual(2, Bandit([1.0, 2.0, 6.0]).optimal_action())


if __name__ == '__main__':
    unittest.main()
