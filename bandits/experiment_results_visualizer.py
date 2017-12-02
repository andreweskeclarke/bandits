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

    def render_regrets(self, results_path, plot_title):
        plt.clf()
        results = None
        directory = os.path.dirname(results_path)
        with open(results_path, 'r') as f:
            results = json.load(f)
        if results is None:
            raise ValueError('No results file! (%s)' % results_path)

        regrets = np.zeros((experiment_results_generator.N_EPISODES_PER_TEST, experiment_results_generator.EPISODE_LENGTH))
        i = 0
        for key, experiment_results in results.items():
            actions = np.array(experiment_results['action'])
            optimal_actions = np.array(experiment_results['optimal_action'])
            regrets[i] = np.not_equal(actions, optimal_actions)
            i += 1

        regrets = np.array(regrets)
        ts_plot = sns.tsplot(data=regrets, ci=[68,95], color='m')
        axes = ts_plot.axes
        axes.set_ylim(0,1)
        axes.set_title(plot_title)
        axes.set_ylabel('% Regret')
        axes.set_xlabel('nth step')

        regrets_plot_path = os.path.join(directory, 'regrets_plot.png')
        ts_plot.get_figure().savefig(regrets_plot_path)
        # ts_plot.show()


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
