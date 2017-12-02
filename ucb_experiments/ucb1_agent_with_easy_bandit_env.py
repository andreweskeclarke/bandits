import os
from bandits import bandit, ucb_agent, experiment_results_generator

if __name__ == '__main__':
    env = [bandit.easy_bandit()]
    agent = ucb_agent.UCBAgent(n_actions=2)
    experiment = ExperimentResultsGenerator()
    experiment.run(env=env, agent=agent)
    experiment.save_results(save_dir=experiment_results_generator.build_experiment_path(__file__))

