import os
import shutil
import unittest
from unittest import mock
from bandits.experiment_results_generator import *


class TestExperimentResultsGenerator(unittest.TestCase):

    def test_build_experiment_path(self):
        py_file = __file__
        self.assertEqual('/home/andrew/src/bandits/test/test_experiment_results_generator/', build_experiment_path(py_file))

        py_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../ucb_experiments/ucb1_agent_with_easy_bandit_env.py')
        self.assertEqual('/home/andrew/src/bandits/ucb_experiments/ucb1_agent_with_easy_bandit_env/', build_experiment_path(py_file))

    def test_get_current_env_for_current_step(self):
        experiment = ExperimentResultsGenerator()
        self.assertEqual('A', experiment._get_env_for_current_step(step=-1, envs=['A']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=0, envs=['A']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=1, envs=['A']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=EPISODE_LENGTH, envs=['A']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=EPISODE_LENGTH*10, envs=['A']))

        self.assertEqual('A', experiment._get_env_for_current_step(step=-1, envs=['A', 'B']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=0, envs=['A', 'B']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=1, envs=['A', 'B']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=2, envs=['A', 'B']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=46, envs=['A', 'B']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=47, envs=['A', 'B']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=48, envs=['A', 'B']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=49, envs=['A', 'B']))
        self.assertEqual('B', experiment._get_env_for_current_step(step=50, envs=['A', 'B']))
        self.assertEqual('B', experiment._get_env_for_current_step(step=51, envs=['A', 'B']))
        self.assertEqual('B', experiment._get_env_for_current_step(step=EPISODE_LENGTH, envs=['A', 'B']))
        self.assertEqual('B', experiment._get_env_for_current_step(step=EPISODE_LENGTH*10, envs=['A', 'B']))

        self.assertEqual('A', experiment._get_env_for_current_step(step=-1, envs=['A', 'B', 'C']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=0, envs=['A', 'B', 'C']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=1, envs=['A', 'B', 'C']))
        self.assertEqual('A', experiment._get_env_for_current_step(step=33, envs=['A', 'B', 'C']))
        self.assertEqual('B', experiment._get_env_for_current_step(step=34, envs=['A', 'B', 'C']))
        self.assertEqual('B', experiment._get_env_for_current_step(step=65, envs=['A', 'B', 'C']))
        self.assertEqual('B', experiment._get_env_for_current_step(step=66, envs=['A', 'B', 'C']))
        self.assertEqual('C', experiment._get_env_for_current_step(step=67, envs=['A', 'B', 'C']))
        self.assertEqual('C', experiment._get_env_for_current_step(step=100, envs=['A', 'B', 'C']))
        self.assertEqual('C', experiment._get_env_for_current_step(step=EPISODE_LENGTH, envs=['A', 'B', 'C']))
        self.assertEqual('C', experiment._get_env_for_current_step(step=EPISODE_LENGTH*10, envs=['A', 'B', 'C']))

    def test_single_bandit_environment(self):
        n_episodes = 10
        agent = mock.Mock()
        agent.configure_mock(**{
            'handle.return_value': 0
            })
        env = mock.Mock()
        env.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            })
        experiment = ExperimentResultsGenerator()
        experiment.run(agent=agent, env=env, n_episodes=n_episodes)

        self.assertEqual(agent.handle.call_count, EPISODE_LENGTH*n_episodes)
        self.assertEqual(env.step.call_count, EPISODE_LENGTH*n_episodes)
        self.assertEqual(agent.reset.call_count, n_episodes)
        self.assertEqual(env.reset.call_count, n_episodes)

    def test_single_bandit_environment_as_list(self):
        n_episodes = 10
        agent = mock.Mock()
        agent.configure_mock(**{
            'handle.return_value': 0
            })
        env = mock.Mock()
        env.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            })
        experiment = ExperimentResultsGenerator()
        experiment.run(agent=agent, env=[env], n_episodes=n_episodes)

        self.assertEqual(agent.handle.call_count, EPISODE_LENGTH*n_episodes)
        self.assertEqual(env.step.call_count, EPISODE_LENGTH*n_episodes)
        self.assertEqual(agent.reset.call_count, n_episodes)
        self.assertEqual(env.reset.call_count, n_episodes)

    def test_two_bandit_environment(self):
        n_episodes = 10
        agent = mock.Mock()
        agent.configure_mock(**{
            'handle.return_value': 0
            })
        env1 = mock.Mock()
        env1.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            })
        env2 = mock.Mock()
        env2.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            })
        experiment = ExperimentResultsGenerator()
        experiment.run(agent=agent, env=[env1, env2], n_episodes=n_episodes)

        self.assertEqual(agent.handle.call_count, EPISODE_LENGTH*n_episodes)
        self.assertEqual(env1.step.call_count, EPISODE_LENGTH*n_episodes / 2.0)
        self.assertEqual(env2.step.call_count, EPISODE_LENGTH*n_episodes / 2.0)
        self.assertEqual(agent.reset.call_count, n_episodes)
        self.assertEqual(env1.reset.call_count, n_episodes)
        self.assertEqual(env2.reset.call_count, n_episodes)

    def test_three_bandit_environment(self):
        n_episodes = 10
        agent = mock.Mock()
        agent.configure_mock(**{
            'handle.return_value': 0
            })
        env1 = mock.Mock()
        env1.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            })
        env2 = mock.Mock()
        env2.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            })
        env3 = mock.Mock()
        env3.configure_mock(**{
            'step.return_value': (None, 0, False, {}),
            'optimal_action.return_value': 0,
            })
        experiment = ExperimentResultsGenerator()
        experiment.run(agent=agent, env=[env1, env2, env3], n_episodes=n_episodes)

        self.assertEqual(agent.handle.call_count, EPISODE_LENGTH*n_episodes)
        self.assertEqual(env1.step.call_count, 0.34 * EPISODE_LENGTH*n_episodes)
        self.assertEqual(env2.step.call_count, 0.33 * EPISODE_LENGTH*n_episodes)
        self.assertEqual(env3.step.call_count, 0.33 * EPISODE_LENGTH*n_episodes)
        self.assertEqual(agent.reset.call_count, n_episodes)
        self.assertEqual(env1.reset.call_count, n_episodes)
        self.assertEqual(env2.reset.call_count, n_episodes)
        self.assertEqual(env3.reset.call_count, n_episodes)

    def test_results_from_experiment(self):
        n_episodes = 10
        agent = mock.Mock()
        agent.configure_mock(**{
            'handle.return_value': np.int64(0)
            })
        env = mock.Mock()
        env.configure_mock(**{
            'step.return_value': (None, np.float(0.0), False, {}),
            'optimal_action.return_value': np.int64(0),
            })
        experiment = ExperimentResultsGenerator()
        experiment.run(agent=agent, env=env, n_episodes=n_episodes)

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
