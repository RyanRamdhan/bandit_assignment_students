#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bandit environment
Practical for course 'Reinforcement Learning',
Bachelor AI, Leiden University, The Netherlands
2021
By Thomas Moerland
"""
import numpy as np
from BanditEnvironment import BanditEnvironment
from BanditPolicies import EgreedyPolicy, OIPolicy, UCBPolicy
from Helper import LearningCurvePlot, ComparisonPlot, smooth
 

def experiment(n_actions, n_timesteps, n_repetitions, smoothing_window):
    #To Do: Write all your experiment code here
    
    # Assignment 1: e-greedy
    """
    input_policy = EgreedyPolicy(n_actions=10)
    epsilon_list = [0.01, 0.05, 0.1, 0.25]
    mean_return_list = np.zeros(len(epsilon_list))
    for epsilon_value in epsilon_list:
        mean_return_list[epsilon_list.index(epsilon_value)] = run_repetitions(n_timesteps=1000, n_repetitions=500, input_value=epsilon_value, policy=input_policy)
    
    LearningCurvePlot.create_mean_return_plot(epsilon_list, mean_return_list, xlabel='epsilon values', title='The mean return of each epsilon value')
    
    # Assignment 2: Optimistic init
    input_policy = OIPolicy(n_actions=10)
    initial_value_list = [0.1, 0.5, 1.0, 2.0]
    mean_return_list = np.zeros(len(initial_value_list))
    for initial_value in initial_value_list:
        mean_return_list[initial_value_list.index(initial_value)] = run_repetitions(n_timesteps=1000, n_repetitions=500, input_value=initial_value, policy=input_policy)
    
    LearningCurvePlot.create_mean_return_plot(initial_value_list, mean_return_list, xlabel='initial values', title='The mean return of each initial value')
    
    
    # Assignment 3: UCB
    input_policy = UCBPolicy(n_actions=10)
    c_list = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
    mean_return_list = np.zeros(len(c_list))
    for c_value in c_list:
        mean_return_list[c_list.index(c_value)] = run_repetitions(n_timesteps=1000, n_repetitions=500, input_value=c_value, policy=input_policy)
    
    LearningCurvePlot.create_mean_return_plot(c_list, mean_return_list, xlabel='c values', title='The mean return of each c value')
    """
    # Assignment 4: Comparison
    plot = ComparisonPlot(title="My Comparison Plot")
    
    policy = EgreedyPolicy(n_actions=10)
    reward_array = run_repetitions(n_timesteps=1000, n_repetitions=500, input_value=0.05, policy=policy)
    smoothed = smooth(reward_array, smoothing_window)
    plot.add_curve(x=np.arange(n_timesteps), y=smoothed, label="Egreedy policy, epsilon: 0.05")
    
    policy = OIPolicy(n_actions=10)
    reward_array = run_repetitions(n_timesteps=1000, n_repetitions=500, input_value=2.0, policy=policy)
    smoothed = smooth(reward_array, smoothing_window)
    plot.add_curve(x=np.arange(n_timesteps), y=smoothed, label="OI policy, initial value: 2.0")
    
    policy = UCBPolicy(n_actions=10)
    reward_array = run_repetitions(n_timesteps=1000, n_repetitions=500, input_value=0.25, policy=policy)
    smoothed = smooth(reward_array, smoothing_window)
    plot.add_curve(x=np.arange(n_timesteps), y=smoothed, label="UCB policy, c value: 0.25")
    
    plot.save("my_plot.png")
    

def run_repetitions(n_timesteps, n_repetitions, input_value, policy):
    reward_array = np.zeros(n_timesteps)
    mean_return = 0
    for i in range(n_repetitions):
        env = BanditEnvironment(n_actions=10)
        policy = EgreedyPolicy(n_actions=10)
        total_reward = 0
        for n in range(n_timesteps):
            a = policy.select_action(input_value)
            r = env.act(a)
            policy.update(a,r)
            total_reward += r
            reward_array[n] += r
        mean_return += total_reward
        print(total_reward)
    
    mean_return = (1 / (n_repetitions * n_timesteps)) * mean_return
    print("mean return" + str(mean_return))
    for reward in reward_array:
        reward = reward / n_repetitions
        round(reward, 1)
    
    #LearningCurvePlot.create_plot(reward_array, smoothing_window=31)
    
    return reward_array
    

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)