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
        null_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        brain = mock.Mock()
        brain.configure_mock(**{
            'single_prediction.return_value': (np.array([1.0, 0.0, 0.0, 0.0]), None, null_state),
            })
        agent = A3CAgent(4, 5, brain)
        actions = np.zeros(4)
        for i in range(100):
            a = agent.act(observation=null_state)
            actions[a] += 1
        self.assertEqual(100, actions[0])

        brain = mock.Mock()
        brain.configure_mock(**{
            'single_prediction.return_value': (np.array([0.0, 0.0, 0.0, 1.0]), None, null_state),
            })
        agent = A3CAgent(4, 5, brain)
        actions = np.zeros(4)
        for i in range(100):
            a = agent.act(observation=null_state)
            actions[a] += 1
        self.assertEqual(100, actions[3])

        brain = mock.Mock()
        brain.configure_mock(**{
            'single_prediction.return_value': (np.array([1.0, 0.0, 0.0, 0.0], ndmin=2), None, null_state),
            })
        agent = A3CAgent(4, 5, brain)
        actions = np.zeros(4)
        for i in range(100):
            a = agent.act(observation=null_state)
            actions[a] += 1
        self.assertEqual(100, actions[0])

        brain = mock.Mock()
        brain.configure_mock(**{
            'single_prediction.return_value': (np.array([0.0, 0.0, 0.0, 1.0], ndmin=2), None, null_state),
            })
        agent = A3CAgent(4, 5, brain)
        actions = np.zeros(4)
        for i in range(100):
            a = agent.act(observation=null_state)
            actions[a] += 1
        self.assertEqual(100, actions[3])


empty_obs = np.array([0, 0, 0])
obs1 = np.array([0, 1, 1])
obs2 = np.array([1, 0, 0])
obs3 = np.array([0, 1, 1])
action1 = np.array([0, 1])
action2 = np.array([1, 0])
action3 = np.array([0, 1])
action4 = np.array([0, 1])
reward1 = 1
reward2 = 0
reward3 = 1
reward4 = 1
null_lstm_state = np.zeros(3)

class TestA3CAgentPushingTrainingEpisodes(unittest.TestCase):

    def test_push_discounted_rewards_to_brain_with_gamma_half_and_single_multistep_lookahead(self):
        brain = mock.Mock()
        gamma = 0.5
        n_look_ahead = 1
        agent = A3CAgent(2, 3, brain, gamma=gamma, n_look_ahead=n_look_ahead)

        agent.push_to_brain(brain, [
            (None, action1, reward1, obs1, null_lstm_state),
            (obs1, action2, reward2, obs2, null_lstm_state),
            (obs2, action3, reward3, obs3, null_lstm_state),
            (obs3, action4, reward4, None, null_lstm_state),
            ])

        call_args = brain.push_training_episode.call_args_list[0][1]
        self.assertTrue(np.equal(call_args['observation'], np.array([empty_obs, obs1, obs2, obs3])).all())
        self.assertTrue(np.equal(call_args['action'], np.array([action1, action2, action3, action4])).all())
        self.assertTrue(np.equal(call_args['next_observation'], np.array([obs1, obs2, obs3, empty_obs])).all())
        self.assertTrue(np.equal(call_args['mask'], np.array([1, 1, 1, 0], ndmin=2).T).all())
        self.assertTrue(np.equal(call_args['reward'], 
            np.array([reward1, reward2, reward3, reward4], ndmin=2).T).all())
        self.assertTrue(np.equal(call_args['discount'], 
            np.array([gamma**1, gamma**1, gamma**1, gamma**1], ndmin=2).T).all())


    def test_push_discounted_rewards_to_brain_with_gamma_half_and_three_multistep_lookahead(self):
        brain = mock.Mock()
        gamma = 0.5
        n_look_ahead = 3
        agent = A3CAgent(2, 3, brain, gamma=gamma, n_look_ahead=n_look_ahead)

        agent.push_to_brain(brain, [
            (None, action1, reward1, obs1, null_lstm_state),
            (obs1, action2, reward2, obs2, null_lstm_state),
            (obs2, action3, reward3, obs3, null_lstm_state),
            (obs3, action4, reward4, None, null_lstm_state),
            ])

        call_args = brain.push_training_episode.call_args_list[0][1]
        self.assertTrue(np.equal(call_args['observation'], np.array([empty_obs, obs1, obs2, obs3])).all())
        self.assertTrue(np.equal(call_args['action'], np.array([action1, action2, action3, action4])).all())
        self.assertTrue(np.equal(call_args['next_observation'], np.array([obs1, obs2, obs3, empty_obs])).all())
        self.assertTrue(np.equal(call_args['mask'], np.array([1, 0, 0, 0], ndmin=2).T).all())
        self.assertTrue(np.equal(call_args['reward'], 
            np.array([
                reward1+gamma*reward2+gamma*gamma*reward3,
                reward2+gamma*reward3+gamma*gamma*reward4,
                reward3+gamma*reward4,
                reward4
                ], ndmin=2).T).all())
        self.assertTrue(np.equal(call_args['discount'],
            np.array([gamma**3, gamma**3, gamma**2, gamma**1], ndmin=2).T).all())


    def test_handle_transitions_with_end_of_episode(self):
        brain = mock.Mock()
        gamma = 0.5
        n_look_ahead = 3
        agent = A3CAgent(2, 3, brain, gamma=gamma, n_look_ahead=n_look_ahead)
        agent.state_h = null_lstm_state

        agent.handle_transition(observation=None, action=np.argmax(action1), reward=reward1, next_observation=obs1, done=False)
        agent.handle_transition(observation=obs1, action=np.argmax(action2), reward=reward2, next_observation=obs2, done=False)
        agent.handle_transition(observation=obs2, action=np.argmax(action3), reward=reward3, next_observation=obs3, done=False)
        brain.push_training_episode.assert_not_called()
        agent.handle_transition(observation=obs3, action=np.argmax(action4), reward=reward4, next_observation=None, done=True)
        brain.push_training_episode.assert_called()

        call_args = brain.push_training_episode.call_args_list[0][1]
        self.assertTrue(np.equal(call_args['observation'], np.array([empty_obs, obs1, obs2, obs3])).all())
        self.assertTrue(np.equal(call_args['action'], np.array([action1, action2, action3, action4])).all())
        self.assertTrue(np.equal(call_args['next_observation'], np.array([obs1, obs2, obs3, empty_obs])).all())
        self.assertTrue(np.equal(call_args['mask'], np.array([1, 0, 0, 0], ndmin=2).T).all())
        self.assertTrue(np.equal(call_args['reward'], 
            np.array([
                reward1+gamma*reward2+gamma*gamma*reward3,
                reward2+gamma*reward3+gamma*gamma*reward4,
                reward3+gamma*reward4,
                reward4
                ], ndmin=2).T).all())
        self.assertTrue(np.equal(call_args['discount'],
            np.array([gamma**3, gamma**3, gamma**2, gamma**1], ndmin=2).T).all())


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

