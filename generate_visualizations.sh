#! /bin/bash
set -e

source activate bandits

echo "Running tests first to ensure nothing is broken..."
bash run_test.sh

echo "Generating visualizations for UCB agent with single bandit environment..."
DISPLAY=:0 PYTHONPATH=`pwd` python bandits/experiment_results_visualizer.py experiments/ucb_single_bandit_experiments/*/results.json

echo "Generating visualizations for UCB agent with two bandit enviroment..."
DISPLAY=:0 PYTHONPATH=`pwd` python bandits/experiment_results_visualizer.py experiments/ucb_two_bandit_experiments/*/results.json

