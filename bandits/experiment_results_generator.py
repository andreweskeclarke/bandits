import re
import os
import json
import math
import numpy as np

N_EPISODES_PER_TEST=100
EPISODE_LENGTH=200


def build_experiment_path(py_file):
    base_directory = os.path.dirname(os.path.realpath(py_file))
    filename = os.path.basename(os.path.realpath(py_file))
    experiment_directory = re.sub('.py', '/', filename)
    path = os.path.join(base_directory, experiment_directory)
    return path


class ExperimentResultsGenerator(object):
    def  __init__(self):
        pass

    def run(self, agent, env, n_episodes=N_EPISODES_PER_TEST):
        self.results = {}
        for i_episode in range(n_episodes):
            episode_results = self.run_one_episode(agent, env)
            self.results[i_episode] = episode_results
        return self.results

    def run_one_episode(self, agent, env):
        action = None
        observation = None
        next_observation = None
        reward = 0
        done = False
        info = {}

        agent.reset()
        next_observation = env.reset()

        episode_results = {
                'action': list(),
                'reward': list(),
                'optimal_action': list(),
                }
        while not done:
            observation = next_observation
            action = agent.act(observation=observation)
            optimal_action = env.optimal_action()
            next_observation, reward, done, info = env.step(action)
            agent.handle_transition(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_observation)

            episode_results['action'].append(action)
            episode_results['reward'].append(reward)
            episode_results['optimal_action'].append(optimal_action)
            env.render()
        return episode_results

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
            return super(NumpyEncoder, self).default(obj)

