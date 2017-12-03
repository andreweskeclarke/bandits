import os
from bandits import bandit, ucb_agent, experiment_results_generator

if __name__ == '__main__':
    env = [bandit.fixed_bandit()]
    env = bandit.MultiBandit(env, episode_length=experiment_results_generator.EPISODE_LENGTH)
    agent = ucb_agent.UCB1Agent(n_actions=env.n_actions())
    experiment = experiment_results_generator.ExperimentResultsGenerator()
    experiment.run(env=env, agent=agent)
    experiment.save_results(save_dir=experiment_results_generator.build_experiment_path(__file__))

