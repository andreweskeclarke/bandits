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

    def test_reset_returns_none_state(self):
        probs = [0.0, 1.0, 0.0, 0.0]
        bandit = Bandit(probs)

        first_state = bandit.reset()
        self.assertEqual(None, first_state)

    def test_finite_length_episode(self):
        probs = [0.0, 1.0, 0.0, 0.0]
        bandit = Bandit(probs, episode_length=10)
        for i in range(9):
            observation, reward, done, info = bandit.step(0)
            self.assertFalse(done)
            self.assertTrue(observation is not None)
        observation, reward, done, info = bandit.step(0)
        self.assertTrue(done)
        self.assertEqual(None, observation)


class TestMultiBandit(unittest.TestCase):

    def test_get_env_for_step(self):
        multi_bandit_env = MultiBandit(['X', 'Y', 'Z'], episode_length=np.inf)
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=-1, envs=['A'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=0, envs=['A'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=1, envs=['A'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=100, envs=['A'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=1000, envs=['A'], episode_length=100))

        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=-1, envs=['A', 'B'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=0, envs=['A', 'B'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=1, envs=['A', 'B'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=2, envs=['A', 'B'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=46, envs=['A', 'B'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=47, envs=['A', 'B'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=48, envs=['A', 'B'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=49, envs=['A', 'B'], episode_length=100))
        self.assertEqual('B', multi_bandit_env._get_env_for_step(step=50, envs=['A', 'B'], episode_length=100))
        self.assertEqual('B', multi_bandit_env._get_env_for_step(step=51, envs=['A', 'B'], episode_length=100))
        self.assertEqual('B', multi_bandit_env._get_env_for_step(step=100, envs=['A', 'B'], episode_length=100))
        self.assertEqual('B', multi_bandit_env._get_env_for_step(step=1000, envs=['A', 'B'], episode_length=100))

        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=-1, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=0, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=1, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('A', multi_bandit_env._get_env_for_step(step=33, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('B', multi_bandit_env._get_env_for_step(step=34, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('B', multi_bandit_env._get_env_for_step(step=65, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('B', multi_bandit_env._get_env_for_step(step=66, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('C', multi_bandit_env._get_env_for_step(step=67, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('C', multi_bandit_env._get_env_for_step(step=100, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('C', multi_bandit_env._get_env_for_step(step=100, envs=['A', 'B', 'C'], episode_length=100))
        self.assertEqual('C', multi_bandit_env._get_env_for_step(step=1000, envs=['A', 'B', 'C'], episode_length=100))


    def test_stepping_through_one_episode_and_resetting(self):
        bandit1 = Bandit([1.0, 0.0])
        bandit2 = Bandit([0.0, 1.0])

        multi_bandit = MultiBandit([bandit1, bandit2], episode_length=4)

        o, r, d, i = multi_bandit.step(0)
        self.assertEqual(r, 1)
        self.assertFalse(d)
        self.assertTrue(np.equal(o, np.array([0, 1])).all())
        o, r, d, i = multi_bandit.step(0)
        self.assertEqual(r, 1)
        self.assertFalse(d)
        self.assertTrue(np.equal(o, np.array([0, 1])).all())
        o, r, d, i = multi_bandit.step(0)
        self.assertEqual(r, 0)
        self.assertFalse(d)
        self.assertTrue(np.equal(o, np.array([0, 0])).all())
        o, r, d, i = multi_bandit.step(0)
        self.assertEqual(r, 0)
        self.assertTrue(d)
        self.assertTrue(o is None)

        with self.assertRaises(ValueError):
            multi_bandit.step(0)

        o = multi_bandit.reset()
        self.assertTrue(o is None)

        # Second time round the probabilities have been shuffled
        o, r, d, i = multi_bandit.step(0)
        self.assertFalse(d)
        o, r, d, i = multi_bandit.step(0)
        self.assertFalse(d)
        o, r, d, i = multi_bandit.step(0)
        self.assertFalse(d)
        o, r, d, i = multi_bandit.step(0)
        self.assertTrue(d)
        self.assertTrue(o is None)

    def test_n_actions(self):
        bandit1 = Bandit([1.0, 0.0])
        bandit2 = Bandit([0.0, 1.0, 0.0, 0.0])

        multi_bandit = MultiBandit([bandit1, bandit2], episode_length=4)

        self.assertEqual(2, multi_bandit.n_actions())
        multi_bandit.step(0)
        self.assertEqual(2, multi_bandit.n_actions())
        multi_bandit.step(0)
        self.assertEqual(4, multi_bandit.n_actions())
        multi_bandit.step(0)
        self.assertEqual(4, multi_bandit.n_actions())
        multi_bandit.step(0)
        self.assertEqual(4, multi_bandit.n_actions())

    def test_optimal_action(self):
        bandit1 = Bandit([1.0, 0.0])
        bandit2 = Bandit([0.0, 1.0, 0.0, 0.0])

        multi_bandit = MultiBandit([bandit1, bandit2], episode_length=4)

        self.assertEqual(0, multi_bandit.optimal_action())
        multi_bandit.step(0)
        self.assertEqual(0, multi_bandit.optimal_action())
        multi_bandit.step(0)
        self.assertEqual(1, multi_bandit.optimal_action())
        multi_bandit.step(0)
        self.assertEqual(1, multi_bandit.optimal_action())
        multi_bandit.step(0)
        self.assertEqual(1, multi_bandit.optimal_action())


if __name__ == '__main__':
    unittest.main()
