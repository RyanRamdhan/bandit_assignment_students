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
        #Initialize nuber of actions, Q(a), and n(a)
        self.n_actions = n_actions
        self.Q_a = np.zeros(n_actions)
        self.n_a = np.zeros(n_actions)
        self.probability_array = np.zeros(n_actions)
        
    def select_action(self, epsilon, t):
        # TO DO: Add own code
        """if self.Q_a.all() == 0:
            for element in self.probability_array:
                element = 1.0/self.n_actions
            try:    
                return np.random.choice(self.n_actions, self.probability_array) 
            except TypeError as e:
                print("Error occurred:", e)
                print("Float causing the error:", self.probability_array[np.where(np.isnan(self.probability_array))])
        """
        #Select action bases on e-greedy strategy
        for n in range(self.n_actions):
            if self.Q_a[n] == np.argmax(self.Q_a):
                #give the optimal action probability of 1 - epsilon
                self.Q_a[n] = 1 - epsilon
            else:
                #give low probabilities to actions that are not optimal
                self.Q_a[n] = epsilon / (self.n_actions - 1)
    
        #execute max action  
        return np.argmax(self.Q_a)  
    
    def update(self,a,r):
        self.n_a[a] += 1
        #update Q(a) based on observed rewards
        self.Q_a[a] += (1/self.n_a[a]) * (r - self.Q_a[a])
        

class OIPolicy:

    def __init__(self, n_actions=10, initial_value=2.0, learning_rate=0.1):
        #Initialize nuber of actions, Q(a), and n(a) and learning rate
        self.n_actions = n_actions
        self.Q_a = np.zeros(n_actions)
        self.learning_rate = learning_rate
        #give each element in Q(a) an initial value
        for n in range(n_actions):
            self.Q_a[n] = initial_value
        
        
    def select_action(self, input_value, t):
        #select action based on optimistic initialization
        for n in range(self.n_actions):
            if self.Q_a[n] == np.argmax(self.Q_a):
                self.Q_a[n] = 1
            else:
                self.Q_a[n] = 0
                
        return np.argmax(self.Q_a)
        
    def update(self,a,r):
        #update Q(a) using the learning rate
        self.Q_a[a] += self.learning_rate * (r - self.Q_a[a])
        

class UCBPolicy:

    def __init__(self, n_actions=10):
        #Initialize nuber of actions, Q(a), and n(a)
        self.n_actions = n_actions
        self.Q_a = np.zeros(n_actions)
        self.n_a = np.zeros(n_actions)
    
    def select_action(self, c, t):
        #treat action as infinity when n(a) = 0
        for n in range(self.n_actions):
            if self.n_a[n] == 0:
                self.Q_a[n] = np.inf
                return np.argmax(self.Q_a)
            
            #calculate the UCB's   
            values = self.Q_a + c * np.sqrt(np.log(t) / self.n_a)   
             
            #select action based on UCB strategy
            if self.Q_a[n] == np.argmax(values):
                self.Q_a[n] = 1
            else:
                self.Q_a[n] = 0
        
        return np.argmax(self.Q_a)
        
    def update(self,a,r):
        self.n_a[a] += 1
        #Update Q(a) based on observed reward
        if self.n_a[a] != 0:
            self.Q_a[a] += (1/self.n_a[a]) * (r - self.Q_a[a])
    
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
