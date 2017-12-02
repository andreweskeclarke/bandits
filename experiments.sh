#! /bin/bash
set -e

source activate bandits

echo "Running tests first to ensure nothing is broken..."
bash run_test.sh

echo "Running UCB experiments..."
for f in ucb_experiments/[^_]*.py; do 
    echo ""
	echo -n "Running experiment defined in: $f"
	time PYTHONPATH=`pwd` python "$f"
done

