from bandits.a3c_agent import *
import unittest
from unittest import mock
import numpy as np

class TestOptimizer(unittest.TestCase):

    def test_run(self):
        stop_signal = threading.Event()
        stop_signal.clear()
        brain = mock.Mock()
        brain.optimize.side_effect = lambda: stop_signal.set()
        optimizer = Optimizer(brain, stop_signal)
        optimizer.run()
        brain.optimize.assert_called_once()


    def test_parallel_run(self):
        stop_signal = threading.Event()
        stop_signal.clear()
        brain1 = mock.Mock()
        brain2 = mock.Mock()
        optimizer1 = Optimizer(brain1, stop_signal)
        optimizer2 = Optimizer(brain2, stop_signal)
        optimizer1.start()
        optimizer2.start()

        stop_signal.set()
        optimizer1.join()
        optimizer2.join()

        brain1.optimize.assert_called()
        brain2.optimize.assert_called()


class TestA3CAgent(unittest.TestCase):

    def test_act_deterministically_greedy(self):
        brain = mock.Mock()
        brain.configure_mock(**{
            'predict_action.return_value': np.array([1.0, 0.0, 0.0, 0.0]),
            })
        agent = A3CAgent(4, brain)
        actions = np.zeros(4)
        for i in range(100):
            a = agent.act(observation=np.array([0.0, 0.0]))
            actions[a] += 1
        self.assertEqual(100, actions[0])

        brain = mock.Mock()
        brain.configure_mock(**{
            'predict_action.return_value': np.array([0.0, 0.0, 0.0, 1.0]),
            })
        agent = A3CAgent(4, brain)
        actions = np.zeros(4)
        for i in range(100):
            a = agent.act(observation=np.array([0.0, 0.0]))
            actions[a] += 1
        self.assertEqual(100, actions[3])


    def test_act_by_fully_exploring(self):
        brain = mock.Mock()
        brain.configure_mock(**{
            'predict_action.return_value': np.array([1.0, 0.0, 0.0, 0.0]),
            })
        agent = A3CAgent(4, brain, epsilon=1.0)
        actions = np.zeros(4)
        for i in range(100):
            a = agent.act(observation=np.array([0.0, 0.0]))
            actions[a] += 1

        self.assertTrue((actions > 10).all())

    def test_push_discounted_rewards_to_brain_with_gamma_1(self):
        brain = mock.Mock()
        agent = A3CAgent(2, brain, gamma=1)
        obs1 = np.array([1, 1])
        obs2 = np.array([0, 1])
        obs3 = np.array([1, 0])
        action1 = np.array([0, 1])
        action2 = np.array([1, 0])
        action3 = np.array([0, 1])
        reward1 = 1
        reward2 = 1
        reward3 = 1

        agent.push_to_brain(brain, [
            (obs1, action1, reward1, obs2),
            (obs2, action2, reward2, obs3),
            (obs3, action3, reward3, None),
            ])

        self.assertTrue(np.equal(obs1, brain.push_training_example.call_args_list[0][1]['observation']).all())
        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[0][1]['next_observation']).all())
        self.assertTrue(np.equal(action1, brain.push_training_example.call_args_list[0][1]['action']).all())
        self.assertEqual(3.0, brain.push_training_example.call_args_list[0][1]['reward'])

        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[1][1]['observation']).all())
        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[1][1]['next_observation']).all())
        self.assertTrue(np.equal(action2, brain.push_training_example.call_args_list[1][1]['action']).all())
        self.assertEqual(2.0, brain.push_training_example.call_args_list[1][1]['reward'])

        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[2][1]['observation']).all())
        self.assertTrue(np.equal(agent._none_state(), brain.push_training_example.call_args_list[2][1]['next_observation']).all())
        self.assertTrue(np.equal(action3, brain.push_training_example.call_args_list[2][1]['action']).all())
        self.assertEqual(1.0, brain.push_training_example.call_args_list[2][1]['reward'])

    def test_push_discounted_rewards_to_brain_with_gamma_half(self):
        brain = mock.Mock()
        agent = A3CAgent(2, brain, gamma=0.5)
        obs1 = np.array([1, 1])
        obs2 = np.array([0, 1])
        obs3 = np.array([1, 0])
        action1 = np.array([0, 1])
        action2 = np.array([1, 0])
        action3 = np.array([0, 1])
        reward1 = 1
        reward2 = 1
        reward3 = 1

        agent.push_to_brain(brain, [
            (obs1, action1, reward1, obs2),
            (obs2, action2, reward2, obs3),
            (obs3, action3, reward3, None),
            ])

        self.assertTrue(np.equal(obs1, brain.push_training_example.call_args_list[0][1]['observation']).all())
        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[0][1]['next_observation']).all())
        self.assertTrue(np.equal(action1, brain.push_training_example.call_args_list[0][1]['action']).all())
        self.assertEqual(1.75, brain.push_training_example.call_args_list[0][1]['reward'])

        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[1][1]['observation']).all())
        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[1][1]['next_observation']).all())
        self.assertTrue(np.equal(action2, brain.push_training_example.call_args_list[1][1]['action']).all())
        self.assertEqual(1.5, brain.push_training_example.call_args_list[1][1]['reward'])

        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[2][1]['observation']).all())
        self.assertTrue(np.equal(agent._none_state(), brain.push_training_example.call_args_list[2][1]['next_observation']).all())
        self.assertTrue(np.equal(action3, brain.push_training_example.call_args_list[2][1]['action']).all())
        self.assertEqual(1.0, brain.push_training_example.call_args_list[2][1]['reward'])

    def test_handle_transitions_with_end_of_episode(self):
        brain = mock.Mock()
        agent = A3CAgent(2, brain, gamma=0.5)
        obs1 = np.array([1, 1])
        obs2 = np.array([0, 1])
        obs3 = np.array([1, 0])
        action1 = 1
        action2 = 0
        action3 = 1
        expected_action1 = np.array([0, 1])
        expected_action2 = np.array([1, 0])
        expected_action3 = np.array([0, 1])
        reward1 = 1
        reward2 = 1
        reward3 = 1

        agent.handle_transition(observation=obs1, action=action1, reward=reward1, next_observation=obs2)
        agent.handle_transition(observation=obs2, action=action2, reward=reward2, next_observation=obs3)
        brain.push_training_example.assert_not_called()
        agent.handle_transition(observation=obs3, action=action3, reward=reward3, next_observation=None)

        self.assertTrue(np.equal(obs1, brain.push_training_example.call_args_list[0][1]['observation']).all())
        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[0][1]['next_observation']).all())
        self.assertTrue(np.equal(expected_action1, brain.push_training_example.call_args_list[0][1]['action']).all())
        self.assertEqual(1.75, brain.push_training_example.call_args_list[0][1]['reward'])

        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[1][1]['observation']).all())
        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[1][1]['next_observation']).all())
        self.assertTrue(np.equal(expected_action2, brain.push_training_example.call_args_list[1][1]['action']).all())
        self.assertEqual(1.5, brain.push_training_example.call_args_list[1][1]['reward'])

        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[2][1]['observation']).all())
        self.assertTrue(np.equal(agent._none_state(), brain.push_training_example.call_args_list[2][1]['next_observation']).all())
        self.assertTrue(np.equal(expected_action3, brain.push_training_example.call_args_list[2][1]['action']).all())
        self.assertEqual(1.0, brain.push_training_example.call_args_list[2][1]['reward'])


if __name__ == '__main__':
    unittest.main()
