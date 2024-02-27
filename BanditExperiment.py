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
    run_repetitions(n_timesteps=1000, n_repetitions=500)
    
    # Assignment 2: Optimistic init
    
    # Assignment 3: UCB
    
    # Assignment 4: Comparison
    
    pass

def run_repetitions(n_timesteps, n_repetitions):
    for i in range(n_repetitions):
        BanditEnvironment(EgreedyPolicy())
        for n in range(n_timesteps):
             a = BanditEnvironment(EgreedyPolicy().select_action(epsilon=0.1))
             r = BanditEnvironment(EgreedyPolicy().Q_a[a])
             BanditEnvironment(EgreedyPolicy().update(a,r))
             
    

if __name__ == '__main__':
    # experiment settings
    n_actions = 10
    n_repetitions = 500
    n_timesteps = 1000
    smoothing_window = 31
    
    experiment(n_actions=n_actions,n_timesteps=n_timesteps,
               n_repetitions=n_repetitions,smoothing_window=smoothing_window)