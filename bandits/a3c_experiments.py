import os
import time
import threading
import numpy as np

from bandits import a3c_agent, a3c_brain, bandit, experiment_results_generator


def run_testing(
        brain,
        env_generator,
        agent_generator,
        save_dir,
        n_episodes=experiment_results_generator.N_EPISODES_PER_TEST):

    env, n_actions, n_inputs = env_generator()
    agent = agent_generator(brain, n_actions)
    experiment = experiment_results_generator.ExperimentResultsGenerator()
    experiment.run(env=env, agent=agent, n_episodes=n_episodes)
    experiment.save_results(save_dir=save_dir)
    return experiment.results


def run_training(
        brain,
        brain_generator,
        env_generator,
        agent_generator,
        training_time,
        n_optimizers,
        n_runners,
        thread_delay=0.001):

    stop_signal = threading.Event()
    stop_signal.clear()
    runners = []
    for i in range(n_runners):
        env, n_actions, n_inputs = env_generator()
        brain = brain if brain is not None else a3c_brain.A3CBrain(n_actions, n_inputs)
        agent = agent_generator(brain, n_actions)
        runners.append(a3c_agent.AsynchRunner(stop_signal, agent, env, thread_delay=thread_delay))

    optimizers = [a3c_agent.AsynchOptimizer(brain, stop_signal) for i in range(n_optimizers)]

    for o in optimizers:
        o.start()
    for r in runners:
        r.start()
    time.sleep(training_time)
    stop_signal.set()
    for o in optimizers:
        o.join()
    for r in runners:
        r.join()

    print('\tCompleted training, encountered %s episodes, trained %s times' %
            (sum([r._total_episodes for r in runners]), brain._n_optimize_runs))
    return brain


def summarize_results(results):
    rewards = np.zeros(experiment_results_generator.EPISODE_LENGTH)
    for k in results.keys():
        rewards = rewards + results[k]['reward']
    rewards = rewards / len(results.items())
    return np.mean(rewards), np.std(rewards)


