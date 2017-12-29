import os
import threading
import time
import numpy as np

from bandits import a3c_agent, a3c_brain, bandit, experiment_results_generator, a3c_experiments


if __name__ == '__main__':
    training_time = 5
    thread_delay=0.0001
    n_optimizers = 8
    n_runners = 8
    gamma = 0.5
    brain = None

    def generate_env():
        env = [bandit.fixed_bandit()]
        env[0]._shuffle_probs = False
        n_actions = env[0].n_actions()
        n_inputs = env[0].n_inputs()
        env = bandit.MultiBandit(env, episode_length=experiment_results_generator.EPISODE_LENGTH)
        return env, n_actions, n_inputs

    def generate_agent(brain, n_actions):
        return a3c_agent.A3CAgent(
                n_actions,
                brain.n_inputs,
                brain,
                thread_delay=thread_delay,
                gamma=gamma)

    def generate_brain(n_actions, n_inputs):
        return a3c_brain.A3CBrain(
                n_actions,
                n_inputs,
                n_timesteps=experiment_results_generator.EPISODE_LENGTH,
                model_name='GRU_MODEL',
                batch_size=8,
                gamma=gamma)

    for i in range(20):
        t = training_time if i > 0 else 0
        brain, n_episodes, n_trials = a3c_experiments.run_training(
                brain=brain,
                brain_generator=generate_brain,
                env_generator=generate_env,
                agent_generator=generate_agent,
                training_time=t, 
                n_optimizers=n_optimizers, 
                n_runners=n_runners,
                thread_delay=thread_delay)
        print('Trained for %s episodes contained %s trials' % (n_episodes, n_trials))
        results = a3c_experiments.run_testing(
                brain=brain,
                env_generator=generate_env,
                agent_generator=generate_agent,
                save_dir=experiment_results_generator.build_experiment_path(__file__),
                n_episodes=50)
        average_reward, std_reward = a3c_experiments.summarize_results(results)
        brain.avg_reward = average_reward
        print('After training iteraton %s, average reward in testing is %s (+/-%s)' % (i, average_reward, std_reward))
        if average_reward == 1.0:
            break

    assert average_reward >= 0.99, 'The observed reward (%s) was lower than expected (%s)' % (average_reward, 1.0)
