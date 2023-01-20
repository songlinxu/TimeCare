import time, os
import random
import numpy as np



class Random_Strategy():

    def __init__(self, action_size = 2):
        self.state = [0.0,0.0]
        self.action = 0
        self.action_size = action_size

    def update_state(self, state):
        self.state = state
             
    def random_action(self):
        return np.random.randint(0,self.action_size)
    
    def update_action(self):
        self.action = self.random_action()
       
        return self.action    

