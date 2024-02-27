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

class EgreedyPolicy:
    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        self.Q_a = np.zeros(n_actions)
        self.n_a = np.zeros(n_actions)
        
        
    def select_action(self, epsilon):
        # TO DO: Add own code
        if np.random.rand() > epsilon:
            a = np.argmax(self.Q_a)
        else:
            a = np.random.randint(self.n_actions)
                
        #execute max action         
        return a
        
    def update(self,a,r):
        # TO DO: Add own code
        self.n_a[a] += 1
        self.Q_a[a] += (1/self.n_a[a]) * (r - self.Q_a[a])
        

class OIPolicy:

    def __init__(self, n_actions=10, initial_value=0.0, learning_rate=0.1):
        self.n_actions = n_actions
        self.Q_a = np.zeros(n_actions)
        self.learning_rate = learning_rate
        for n in range(n_actions):
            self.Q_a[n] = initial_value
        
        
    def select_action(self):
        # TO DO: Add own code
        for n in range(self.n_actions):
            if self.Q_a[n] == np.argmax(self.Q_a):
                self.Q_a[n] = 1
            else:
                self.Q_a[n] = 0
                
        return np.argmax(self.Q_a)
        
    def update(self,a,r):
        # TO DO: Add own code
        self.Q_a[a] += self.learning_rate * (r - self.Q_a[a])
        

class UCBPolicy:

    def __init__(self, n_actions=10):
        self.n_actions = n_actions
        # TO DO: Add own code
        pass
    
    def select_action(self, c, t):
        # TO DO: Add own code
        a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        return a
        
    def update(self,a,r):
        # TO DO: Add own code
        pass
    
def test():
    n_actions = 10
    env = BanditEnvironment(n_actions=n_actions) # Initialize environment    
    
    pi = EgreedyPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(epsilon=0.5) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test e-greedy policy with action {}, received reward {}".format(a,r))
    
    pi = OIPolicy(n_actions=n_actions,initial_value=1.0) # Initialize policy
    a = pi.select_action() # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test greedy optimistic initialization policy with action {}, received reward {}".format(a,r))
    
    pi = UCBPolicy(n_actions=n_actions) # Initialize policy
    a = pi.select_action(c=1.0,t=1) # select action
    r = env.act(a) # sample reward
    pi.update(a,r) # update policy
    print("Test UCB policy with action {}, received reward {}".format(a,r))
    
if __name__ == '__main__':
    test()
