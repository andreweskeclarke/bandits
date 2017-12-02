#! /bin/bash
set -e

source activate bandits

echo "Running tests first to ensure nothing is broken..."
bash run_test.sh

echo "Running UCB experiments with single bandit per episode..."
for f in experiments/ucb_single_bandit_experiments/[^_]*.py; do 
    echo ""
	echo -n "Running experiment defined in: $f"
	time PYTHONPATH=`pwd` python "$f"
done

echo "Running UCB experiments with two bandits per episode..."
for f in experiments/ucb_two_bandit_experiments/[^_]*.py; do 
    echo ""
	echo -n "Running experiment defined in: $f"
	time PYTHONPATH=`pwd` python "$f"
done

