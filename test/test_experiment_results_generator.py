import os
import shutil
import unittest
from unittest import mock
from bandits.experiment_results_generator import *
from bandits.bandit import *


class TestExperimentResultsGenerator(unittest.TestCase):

    def test_build_experiment_path(self):
        py_file = __file__
        self.assertEqual('/home/andrew/src/bandits/test/test_experiment_results_generator/', build_experiment_path(py_file))

        py_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../ucb_experiments/ucb1_agent_with_easy_bandit_env.py')
        self.assertEqual('/home/andrew/src/bandits/ucb_experiments/ucb1_agent_with_easy_bandit_env/', build_experiment_path(py_file))

    def test_single_bandit_environment(self):
        n_episodes = 3
        episode_length = 55
        agent = mock.Mock()
        agent.configure_mock(**{
            'act.return_value': 0
            })
        env = Bandit([1.0, 0.0], episode_length=episode_length)
        experiment = ExperimentResultsGenerator()
        experiment.run(
                agent=agent,
                env=env,
                n_episodes=n_episodes)

        self.assertEqual(agent.act.call_count, episode_length*n_episodes)
        self.assertEqual(agent.handle_transition.call_count, episode_length*n_episodes)
        self.assertEqual(agent.reset.call_count, n_episodes)

        # Second to last transition that is pushed to the agent
        self.assertTrue(np.equal(
            agent.handle_transition.call_args_list[-2][1]['observation'],
            np.array([1, 0, 1])).all())
        self.assertEqual(agent.handle_transition.call_args_list[-2][1]['action'], 0)
        self.assertEqual(agent.handle_transition.call_args_list[-2][1]['reward'], 1)
        self.assertTrue(np.equal(
            agent.handle_transition.call_args_list[-2][1]['next_observation'],
            np.array([1, 0, 1])).all())
        self.assertEqual(agent.handle_transition.call_args_list[-2][1]['done'], False)

        # Last transition that is pushed to the agent
        self.assertTrue(np.equal(
            agent.handle_transition.call_args[1]['observation'],
            np.array([1, 0, 1])).all())
        self.assertEqual(agent.handle_transition.call_args[1]['action'], 0)
        self.assertEqual(agent.handle_transition.call_args[1]['reward'], 1)
        self.assertEqual(agent.handle_transition.call_args[1]['next_observation'], None)
        self.assertEqual(agent.handle_transition.call_args[1]['done'], True)

    def test_two_bandit_environment(self):
        n_episodes = 3
        episode_length = 10
        agent = mock.Mock()
        agent.configure_mock(**{
            'act.return_value': 0
            })
        env1 = mock.Mock()
        env1.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            'n_inputs.return_value': 3,
            })
        env2 = mock.Mock()
        env2.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            'n_inputs.return_value': 3,
            })
        experiment = ExperimentResultsGenerator()
        experiment.run(
                agent=agent,
                env=MultiBandit([env1, env2], episode_length=episode_length),
                n_episodes=n_episodes)

        self.assertEqual(agent.act.call_count, episode_length*n_episodes)
        self.assertEqual(agent.handle_transition.call_count, episode_length*n_episodes)
        self.assertEqual(agent.reset.call_count, n_episodes)
        self.assertEqual(env1.step.call_count, episode_length*n_episodes / 2.0)
        self.assertEqual(env2.step.call_count, episode_length*n_episodes / 2.0)
        self.assertEqual(env1.reset.call_count, n_episodes)
        self.assertEqual(env2.reset.call_count, n_episodes)

    def test_four_bandit_environment(self):
        n_episodes = 3
        episode_length = 24
        agent = mock.Mock()
        agent.configure_mock(**{
            'act.return_value': 0
            })
        env1 = mock.Mock()
        env1.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            'n_inputs.return_value': 3,
            })
        env2 = mock.Mock()
        env2.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            'n_inputs.return_value': 3,
            })
        env3 = mock.Mock()
        env3.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            'n_inputs.return_value': 3,
            })
        env4 = mock.Mock()
        env4.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            'n_inputs.return_value': 3,
            })
        experiment = ExperimentResultsGenerator()
        experiment.run(
                agent=agent,
                env=MultiBandit([env1, env2, env3, env4], episode_length=episode_length),
                n_episodes=n_episodes)

        self.assertEqual(agent.act.call_count, episode_length*n_episodes)
        self.assertEqual(agent.handle_transition.call_count, episode_length*n_episodes)
        self.assertEqual(agent.reset.call_count, n_episodes)
        self.assertEqual(env1.step.call_count, 0.25*episode_length*n_episodes)
        self.assertEqual(env2.step.call_count, 0.25*episode_length*n_episodes)
        self.assertEqual(env3.step.call_count, 0.25*episode_length*n_episodes)
        self.assertEqual(env4.step.call_count, 0.25*episode_length*n_episodes)
        self.assertEqual(env1.reset.call_count, n_episodes)
        self.assertEqual(env2.reset.call_count, n_episodes)
        self.assertEqual(env3.reset.call_count, n_episodes)
        self.assertEqual(env4.reset.call_count, n_episodes)

    def test_results_from_experiment(self):
        n_episodes = 3
        episode_length = 10
        agent = mock.Mock()
        agent.configure_mock(**{
            'act.return_value': np.int64(0)
            })
        env = mock.Mock()
        env.configure_mock(**{
            'step.return_value': (None, np.float(0.0), False, {}),
            'optimal_action.return_value': np.int64(0),
            'n_inputs.return_value': 3,
            })
        experiment = ExperimentResultsGenerator()
        experiment.run(
                agent=agent,
                env=MultiBandit([env], episode_length=episode_length),
                n_episodes=n_episodes)

        converted_output = json.dumps(experiment.results, cls=NumpyEncoder)
        # Really just want to ensure this doesnt blow up
        self.assertTrue(converted_output is not None)

class TestExperimentResultsGeneratorOutput(unittest.TestCase):

    def save_dir(self):
        return build_experiment_path(__file__)

    def setUp(self):
        # Make sure there is no saved directory
        assert not os.path.exists(self.save_dir())

    def test_save_results(self):
        experiment = ExperimentResultsGenerator()
        experiment.results = {'my_key': 'my_value'}
        experiment.save_results(self.save_dir())
        with open(os.path.join(self.save_dir(), 'results.json'), 'r') as f:
            results = json.load(f)
            self.assertEqual('my_value', results['my_key'])

    def tearDown(self):
        # Remove the newly created directory
        shutil.rmtree(self.save_dir())


if __name__ == '__main__':
    unittest.main()
