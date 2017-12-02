# Multi Armed Bandits

This repo is for playing around with some ideas in Multi-Armed bandits and LSTMs.

Inspired by a presentation from Thomas Lecat.

# Setup

```sh
conda env create -f environment.yaml
source activate bandits
./run_test.sh
```

# Running Experiments

All experiments will run from a single shell script:

```sh
./experiments.sh
./generate_visualizations.sh
```

Results are saved in JSON files and png files. For example, to view all the output images, I use `feh`:
```sh
feh experiments/*/*/regrets_plot.png
```
