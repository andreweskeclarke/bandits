#! /bin/bash
set -e

source activate bandits

echo "Running tests first to ensure nothing is broken..."
bash run_test.sh

echo "Running UCB visualizations..."
DISPLAY=:0 PYTHONPATH=`pwd` python bandits/experiment_results_visualizer.py ucb_experiments/*/results.json

