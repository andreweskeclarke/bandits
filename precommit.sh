#! /bin/bash
set -e

source activate bandits

echo "Unit tests..."
bash run_test.sh

echo "Functional tests..."
bash functional_tests.sh

echo "UCB experiments..."
bash experiments.sh

