import os
import threading
import time
import numpy as np

from bandits import a3c_agent, a3c_brain, bandit, experiment_results_generator


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
        env, n_actions, n_inputs = generate_env()
        brain = brain if brain is not None else a3c_brain.A3CBrain(n_actions, n_inputs)
        agent = generate_agent(brain, n_actions)
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


def run_testing(
        brain,
        env_generator,
        agent_generator,
        n_episodes=experiment_results_generator.N_EPISODES_PER_TEST):

    env, n_actions, n_inputs = env_generator()
    agent = agent_generator(brain, n_actions)

    experiment = experiment_results_generator.ExperimentResultsGenerator()
    experiment.run(env=env, agent=agent, n_episodes=n_episodes)
    experiment.save_results(save_dir=experiment_results_generator.build_experiment_path(__file__))

    results = experiment.results
    rewards = np.zeros(experiment_results_generator.EPISODE_LENGTH)
    for k in results.keys():
        rewards = rewards + results[k]['reward']
    rewards = rewards / len(results.items())
    return np.mean(rewards), np.std(rewards)

if __name__ == '__main__':
    training_time = 5
    thread_delay=0.0001
    n_optimizers = 8
    n_runners = 8
    brain = None

    def generate_env():
        env = [bandit.fixed_bandit()]
        env[0]._shuffle_probs = False
        n_actions = env[0].n_actions()
        n_inputs = env[0].n_inputs()
        env = bandit.MultiBandit(env, episode_length=experiment_results_generator.EPISODE_LENGTH)
        return env, n_actions, n_inputs

    def generate_agent(brain, n_actions):
        return a3c_agent.A3CAgent(n_actions, brain, thread_delay=thread_delay)

    def generate_brain(n_actions, n_inputs):
        return a3c_brain.A3CBrain(n_actions, n_inputs)

    for i in range(3):
        t = training_time if i > 0 else 0
        brain = run_training(
                brain=brain,
                brain_generator=generate_brain,
                env_generator=generate_env,
                agent_generator=generate_agent,
                training_time=t, 
                n_optimizers=n_optimizers, 
                n_runners=n_runners,
                thread_delay=thread_delay)
        average_reward, std_reward = run_testing(
                brain=brain,
                env_generator=generate_env,
                agent_generator=generate_agent,
                n_episodes=50)
        brain.reset()
        print('After training iteraton %s, average reward in testing is %s (+/-%s)' % (i, average_reward, std_reward))

    assert average_reward >= 0.99, 'The observed reward (%s) was lower than expected (%s)' % (average_reward, 1.0)
