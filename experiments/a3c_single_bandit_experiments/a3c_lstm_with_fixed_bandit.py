import os
import threading
import time
import numpy as np

from bandits import a3c_agent, a3c_brain, bandit, experiment_results_generator, a3c_experiments


if __name__ == '__main__':
    training_time = 30
    thread_delay=0.0001
    n_optimizers = 8
    n_runners = 8
    brain = None
    episode_length = 25

    gamma=0.8
    learning_rate = 1e-3
    batch_size = 32
    coef_value_loss = 0.1
    coef_entropy_loss = 0.05
    n_training_epochs = 20

    n_total_training_episodes = 0
    n_testing_episodes = 100
    i = 0

    def generate_env():
        env = [bandit.easy_bandit()]
        env = bandit.MultiBandit(env, episode_length=episode_length, include_steps=True)
        n_actions = env.n_actions()
        n_inputs = env.n_inputs()
        return env, n_actions, n_inputs

    def generate_agent(brain, n_actions):
        return a3c_agent.A3CAgent(
                n_actions,
                brain,
                thread_delay=thread_delay,
                gamma=gamma)

    def generate_brain(n_actions, n_inputs):
        print('GENERATE NEW BRAIN')
        return a3c_brain.A3CBrain(
                n_actions=n_actions,
                n_inputs=n_inputs,
                n_timesteps=episode_length,
                model_name='LSTM_MODEL',
                batch_size=batch_size,
                coef_value_loss=coef_value_loss,
                coef_entropy_loss=coef_entropy_loss,
                gamma=gamma,
                learning_rate=learning_rate)

    while n_total_training_episodes < 5000000:
        t = training_time if n_total_training_episodes > 0 else 0
        brain, n_episodes, n_trials = a3c_experiments.run_training(
                brain=brain,
                brain_generator=generate_brain,
                env_generator=generate_env,
                agent_generator=generate_agent,
                training_time=t, 
                n_optimizers=n_optimizers, 
                n_runners=n_runners,
                thread_delay=thread_delay)
        n_total_training_episodes += n_episodes

        results = a3c_experiments.run_testing(
                brain=brain,
                env_generator=generate_env,
                agent_generator=generate_agent,
                save_dir=experiment_results_generator.build_experiment_path(__file__),
                n_episodes=n_testing_episodes)
        average_reward, std_reward = a3c_experiments.summarize_results(results, episode_length=episode_length)
        brain.reset()
        brain.avg_reward = average_reward
        print('After training iteraton %s, average reward in testing is %s (+/-%s) (%s total episodes so far)' % (i, average_reward, std_reward, n_total_training_episodes))
        i += 1

    assert average_reward >= 0.99, 'The observed reward (%s) was lower than expected (%s)' % (average_reward, 1.0)
