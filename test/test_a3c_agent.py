from bandits.a3c_agent import *
import unittest
from unittest import mock
import numpy as np

class TestAsynchOptimizer(unittest.TestCase):

    def test_run(self):
        stop_signal = threading.Event()
        stop_signal.clear()
        brain = mock.Mock()
        brain.optimize.side_effect = lambda: stop_signal.set()
        optimizer = AsynchOptimizer(brain, stop_signal)
        optimizer.run()
        brain.optimize.assert_called_once()


    def test_parallel_run(self):
        stop_signal = threading.Event()
        stop_signal.clear()
        brain1 = mock.Mock()
        brain2 = mock.Mock()
        optimizer1 = AsynchOptimizer(brain1, stop_signal)
        optimizer2 = AsynchOptimizer(brain2, stop_signal)
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

        brain = mock.Mock()
        brain.configure_mock(**{
            'predict_action.return_value': np.array([1.0, 0.0, 0.0, 0.0], ndmin=2),
            })
        agent = A3CAgent(4, brain)
        actions = np.zeros(4)
        for i in range(100):
            a = agent.act(observation=np.array([0.0, 0.0]))
            actions[a] += 1
        self.assertEqual(100, actions[0])

        brain = mock.Mock()
        brain.configure_mock(**{
            'predict_action.return_value': np.array([0.0, 0.0, 0.0, 1.0], ndmin=2),
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
        self.assertEqual(1.0, brain.push_training_example.call_args_list[0][1]['discount'])
        self.assertEqual(3.0, brain.push_training_example.call_args_list[0][1]['reward'])

        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[1][1]['observation']).all())
        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[1][1]['next_observation']).all())
        self.assertTrue(np.equal(action2, brain.push_training_example.call_args_list[1][1]['action']).all())
        self.assertEqual(1.0, brain.push_training_example.call_args_list[1][1]['discount'])
        self.assertEqual(2.0, brain.push_training_example.call_args_list[1][1]['reward'])

        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[2][1]['observation']).all())
        self.assertTrue(np.equal(None, brain.push_training_example.call_args_list[2][1]['next_observation']).all())
        self.assertTrue(np.equal(action3, brain.push_training_example.call_args_list[2][1]['action']).all())
        self.assertEqual(1.0, brain.push_training_example.call_args_list[2][1]['discount'])
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
        self.assertEqual(1/8., brain.push_training_example.call_args_list[0][1]['discount'])
        self.assertEqual(1.75, brain.push_training_example.call_args_list[0][1]['reward'])

        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[1][1]['observation']).all())
        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[1][1]['next_observation']).all())
        self.assertTrue(np.equal(action2, brain.push_training_example.call_args_list[1][1]['action']).all())
        self.assertEqual(1/4., brain.push_training_example.call_args_list[1][1]['discount'])
        self.assertEqual(1.5, brain.push_training_example.call_args_list[1][1]['reward'])

        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[2][1]['observation']).all())
        self.assertTrue(np.equal(None, brain.push_training_example.call_args_list[2][1]['next_observation']).all())
        self.assertTrue(np.equal(action3, brain.push_training_example.call_args_list[2][1]['action']).all())
        self.assertEqual(1/2., brain.push_training_example.call_args_list[2][1]['discount'])
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

        agent.handle_transition(observation=obs1, action=action1, reward=reward1, next_observation=obs2, done=False)
        agent.handle_transition(observation=obs2, action=action2, reward=reward2, next_observation=obs3, done=False)
        brain.push_training_example.assert_not_called()
        agent.handle_transition(observation=obs3, action=action3, reward=reward3, next_observation=None, done=True)

        self.assertTrue(np.equal(obs1, brain.push_training_example.call_args_list[0][1]['observation']).all())
        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[0][1]['next_observation']).all())
        self.assertTrue(np.equal(expected_action1, brain.push_training_example.call_args_list[0][1]['action']).all())
        self.assertEqual(1/8., brain.push_training_example.call_args_list[0][1]['discount'])
        self.assertEqual(1.75, brain.push_training_example.call_args_list[0][1]['reward'])

        self.assertTrue(np.equal(obs2, brain.push_training_example.call_args_list[1][1]['observation']).all())
        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[1][1]['next_observation']).all())
        self.assertTrue(np.equal(expected_action2, brain.push_training_example.call_args_list[1][1]['action']).all())
        self.assertEqual(1/4., brain.push_training_example.call_args_list[1][1]['discount'])
        self.assertEqual(1.5, brain.push_training_example.call_args_list[1][1]['reward'])

        self.assertTrue(np.equal(obs3, brain.push_training_example.call_args_list[2][1]['observation']).all())
        self.assertTrue(np.equal(None, brain.push_training_example.call_args_list[2][1]['next_observation']).all())
        self.assertTrue(np.equal(expected_action3, brain.push_training_example.call_args_list[2][1]['action']).all())
        self.assertEqual(1/2., brain.push_training_example.call_args_list[2][1]['discount'])
        self.assertEqual(1.0, brain.push_training_example.call_args_list[2][1]['reward'])


class TestAsynchRunner(unittest.TestCase):

    def test_run_one_episode(self):
        agent = mock.Mock()
        agent.configure_mock(**{
            'act.return_value': -1
            })
        env = mock.Mock()
        env.configure_mock(**{
            'reset.return_value': None,
            'step.return_value': (None, 0, True, {})
            })
        runner = AsynchRunner(threading.Event(), None, None, thread_delay=0.0)
        runner.run_one_episode(agent, env)

        agent.reset.assert_called_once()
        agent.act.assert_called_once()
        agent.handle_transition.assert_called_once_with(observation=None, action=-1, reward=0, next_observation=None, done=True)
        env.step.assert_called_once_with(action=-1)
        env.reset.assert_called_once()

    def test_parallel_run(self):
        stop_signal = threading.Event()
        stop_signal.clear()

        agent1 = mock.Mock()
        agent1.configure_mock(**{
            'act.return_value': -1
            })
        env1 = mock.Mock()
        env1.configure_mock(**{
            'reset.return_value': None,
            'step.return_value': (None, 0, False, {})
            })
        agent2 = mock.Mock()
        agent2.configure_mock(**{
            'act.return_value': -1
            })
        env2 = mock.Mock()
        env2.configure_mock(**{
            'reset.return_value': None,
            'step.return_value': (None, 0, False, {})
            })

        runner1 = AsynchRunner(stop_signal, agent1, env1, thread_delay=0.0)
        runner2 = AsynchRunner(stop_signal, agent2, env2, thread_delay=0.0)
        runner1.start()
        runner2.start()

        stop_signal.set()
        runner1.join()
        runner2.join()

        agent1.reset.assert_called()
        agent1.act.assert_called()
        agent1.handle_transition.assert_called()
        env1.step.assert_called()
        env1.reset.assert_called()

        agent2.reset.assert_called()
        agent2.act.assert_called()
        agent2.handle_transition.assert_called()
        env2.step.assert_called()
        env2.reset.assert_called()



if __name__ == '__main__':
    unittest.main()

