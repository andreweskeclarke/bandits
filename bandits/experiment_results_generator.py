import re
import os
import json
import math
import numpy as np

N_EPISODES_PER_TEST=100
EPISODE_LENGTH=100


def build_experiment_path(py_file):
    base_directory = os.path.dirname(os.path.realpath(py_file))
    filename = os.path.basename(os.path.realpath(py_file))
    experiment_directory = re.sub('.py', '/', filename)
    path = os.path.join(base_directory, experiment_directory)
    return path


class ExperimentResultsGenerator(object):
    def  __init__(self):
        pass

    def _get_env_for_current_step(self, step, envs):
        step = max(0, min(EPISODE_LENGTH-1, step))
        percent_done = float(step) / float(EPISODE_LENGTH)
        current_env_index = math.floor(percent_done * float(len(envs)))
        current_env_index = max(0, min(len(envs)-1, current_env_index))
        return envs[current_env_index]

    def run(self, agent, env, n_episodes=N_EPISODES_PER_TEST):
        if not isinstance(env, list):
            env = [env]

        action = None
        observation = None
        reward = 0
        done = False
        info = {}

        self.results = {}
        for i_episode in range(n_episodes):
            episode_results = {
                    'action': np.zeros(EPISODE_LENGTH).tolist(),
                    'reward': np.zeros(EPISODE_LENGTH).tolist(),
                    'optimal_action': np.zeros(EPISODE_LENGTH).tolist(),
                    }
            for i_step in range(EPISODE_LENGTH):
                current_env = self._get_env_for_current_step(i_step, env) 
                action = agent.handle(action=action, observation=observation, reward=reward, done=done, info=info)
                optimal_action = current_env.optimal_action()
                observation, reward, done, info = current_env.step(action)

                episode_results['action'][i_step] = action
                episode_results['reward'][i_step] = reward
                episode_results['optimal_action'][i_step] = optimal_action

            self.results[i_episode] = episode_results

    def save_results(self, save_dir):
        results_output_path = os.path.join(save_dir, 'results.json')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(results_output_path, 'w') as out:
            json.dump(self.results, out, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

