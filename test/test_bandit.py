import unittest
from bandits.bandit import *

class TestBandit(unittest.TestCase):

    def test_optimal_action(self):
        self.assertEqual(0, Bandit([1.0, 1.0, 1.0]).optimal_action())
        self.assertEqual(0, Bandit([1.0, 0.0, 0.0]).optimal_action())
        self.assertEqual(1, Bandit([0.0, 2.0, 0.0]).optimal_action())
        self.assertEqual(2, Bandit([1.0, 2.0, 6.0]).optimal_action())

    def test_observations_are_previous_action_reward_tuple(self):
        bandit = Bandit([0.0, 1.0, 0.0, 0.0])
        observation, reward, done, info = bandit.step(0.0)
        self.assertTrue(np.equal(np.array([0.0, 0.0]), observation).all())
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        observation, reward, done, info = bandit.step(1.0)
        self.assertTrue(np.equal(np.array([1.0, 1.0]), observation).all())
        self.assertEqual(reward, 1.0)
        self.assertFalse(done)
        observation, reward, done, info = bandit.step(2.0)
        self.assertTrue(np.equal(np.array([2.0, 0.0]), observation).all())
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        observation, reward, done, info = bandit.step(3.0)
        self.assertTrue(np.equal(np.array([3.0, 0.0]), observation).all())
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)

        observation, reward, done, info = bandit.step(0)
        self.assertTrue(np.equal(np.array([0.0, 0.0]), observation).all())
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        observation, reward, done, info = bandit.step(1)
        self.assertTrue(np.equal(np.array([1.0, 1.0]), observation).all())
        self.assertEqual(reward, 1.0)
        self.assertFalse(done)
        observation, reward, done, info = bandit.step(2)
        self.assertTrue(np.equal(np.array([2.0, 0.0]), observation).all())
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        observation, reward, done, info = bandit.step(3)
        self.assertTrue(np.equal(np.array([3.0, 0.0]), observation).all())
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)

    def test_reset_shuffles_probabilities(self):
        probs = [0.0, 1.0, 0.0, 0.0]
        bandit = Bandit(probs)

        self.assertEqual(bandit.n_arms, 4)
        self.assertTrue((probs == bandit.arm_probabilities).all())

        summed_probs = np.zeros(4)
        for i in range(100):
            bandit.reset()
            summed_probs = summed_probs + bandit.arm_probabilities
        self.assertTrue((summed_probs > 2).all())

if __name__ == '__main__':
    unittest.main()
