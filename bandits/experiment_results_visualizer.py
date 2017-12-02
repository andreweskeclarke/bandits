import os
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import experiment_results_generator

class ExperimentResultsVisualization(object):
    def __init__(self):
        pass

    def _render_timeseries(self, data, ylabel, plot_title, save_path):
        plt.clf()
        data = np.array(data)
        ts_plot = sns.tsplot(data=data, ci=[68,95], color='m')
        axes = ts_plot.axes
        axes.set_ylim(0,1)
        axes.set_title(plot_title)
        axes.set_ylabel(ylabel)
        axes.set_xlabel('nth step')
        print('Saving %s' % save_path)
        ts_plot.get_figure().savefig(save_path)

    def _load_results(self, results_path):
        results = None
        directory = os.path.dirname(results_path)
        with open(results_path, 'r') as f:
            results = json.load(f)
        if results is None:
            raise ValueError('No results file! (%s)' % results_path)
        return results

    def _save_path(self, results_path, fig_filename):
        directory = os.path.dirname(results_path)
        save_path = os.path.join(directory, fig_filename)
        return save_path

    def render_regrets(self, results_path, plot_title):
        results = self._load_results(results_path)
        regrets = np.zeros((experiment_results_generator.N_EPISODES_PER_TEST, experiment_results_generator.EPISODE_LENGTH))
        i = 0
        for key, experiment_results in results.items():
            actions = np.array(experiment_results['action'])
            optimal_actions = np.array(experiment_results['optimal_action'])
            regrets[i] = np.not_equal(actions, optimal_actions)
            i += 1
        self._render_timeseries(
                regrets, 
                '% Regret', 
                plot_title, 
                self._save_path(results_path, 'regrets_plot.png'))

    def render_rewards(self, results_path, plot_title):
        results = self._load_results(results_path)
        rewards = np.zeros((experiment_results_generator.N_EPISODES_PER_TEST, experiment_results_generator.EPISODE_LENGTH))
        i = 0
        for key, experiment_results in results.items():
            rewards[i] = experiment_results['reward']
            i += 1
        self._render_timeseries(
                rewards, 
                'Reward', 
                plot_title, 
                self._save_path(results_path, 'rewards_plot.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('paths', 
            type=str, 
            nargs='+',
            help='The paths to the experiment results we want to plot')
    args = parser.parse_args()
    viz = ExperimentResultsVisualization()
    for results_path in args.paths:
        plot_title = os.path.dirname(results_path)
        plot_title = plot_title.split('/')[-1]
        plot_title = re.sub('_', ' ', plot_title)
        viz.render_regrets(results_path, plot_title)
        viz.render_rewards(results_path, plot_title)
