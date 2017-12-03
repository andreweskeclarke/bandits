import os
import threading
import time
import numpy as np

from bandits import a3c_agent, a3c_brain, bandit, experiment_results_generator

THREAD_DELAY=0.00001

def run_training(brain, training_time, n_optimizers, n_runners):
    stop_signal = threading.Event()
    stop_signal.clear()
    runners = []
    for i in range(n_runners):
        env = [bandit.fixed_bandit()]
        env[0]._shuffle_probs = False
        n_actions = env[0].n_actions()
        n_inputs = env[0].n_inputs()
        env = bandit.MultiBandit(env, episode_length=experiment_results_generator.EPISODE_LENGTH)
        if brain is None:
            brain = a3c_brain.A3CBrain(n_actions, n_inputs)
        agent = a3c_agent.A3CAgent(n_actions, brain, thread_delay=THREAD_DELAY)
        runners.append(a3c_agent.AsynchRunner(stop_signal, agent, env, thread_delay=THREAD_DELAY))

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


def run_testing(brain):
    env = [bandit.fixed_bandit()]
    env[0]._shuffle_probs = False
    env = bandit.MultiBandit(env, episode_length=experiment_results_generator.EPISODE_LENGTH)
    n_actions = env.n_actions()
    n_inputs = env.n_inputs()
    agent = a3c_agent.A3CAgent(n_actions=n_actions, brain=brain)
    experiment = experiment_results_generator.ExperimentResultsGenerator()
    experiment.run(env=env, agent=agent, n_episodes=20)
    experiment.save_results(save_dir=experiment_results_generator.build_experiment_path(__file__))

    results = experiment.results
    rewards = np.zeros(experiment_results_generator.EPISODE_LENGTH)
    for k in results.keys():
        rewards = rewards + results[k]['reward']
    rewards = rewards / len(results.items())
    return np.mean(rewards)

if __name__ == '__main__':
    training_time = 5
    n_optimizers = 8
    n_runners = 8
    brain = None

    for i in range(5):
        t = training_time if i > 0 else 0
        brain = run_training(brain=brain, training_time=t, n_optimizers=n_optimizers, n_runners=n_runners)
        average_reward = run_testing(brain=brain)
        brain.reset()
        print('After training iteraton %s, average reward in testing is %s' % (i, average_reward))

    assert average_reward >= 0.99, 'The observed reward (%s) was lower than expected (%s)' % (average_reward, 1.0)
