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
    epsilon_list = [0.01, 0.05, 0.1, 0.25]
    mean_return_list = np.zeros(len(epsilon_list))
    for epsilon_value in epsilon_list:
        mean_return_list[epsilon_list.index(epsilon_value)] = run_repetitions(n_timesteps=1000, n_repetitions=500, epsilon=epsilon_value)
    
    LearningCurvePlot.create_mean_return_plot(epsilon_list, mean_return_list, xlabel='epsilon values', title='The mean return of each epsilon value')
    
    
    # Assignment 2: Optimistic init
    
    # Assignment 3: UCB
    
    # Assignment 4: Comparison
    
    pass

def run_repetitions(n_timesteps, n_repetitions, epsilon):
    reward_array = np.zeros(n_timesteps)
    mean_return = 0
    for i in range(n_repetitions):
        env = BanditEnvironment(n_actions=10)
        policy = EgreedyPolicy(n_actions=10)
        total_reward = 0
        for n in range(n_timesteps):
            a = policy.select_action(epsilon)
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
    
    LearningCurvePlot.create_plot(reward_array, smoothing_window=31)
    
    return mean_return
    

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)