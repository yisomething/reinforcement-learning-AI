"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
from tile3 import*


class Agent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)
    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.epsilon =0
        self.gamma = 1.0
        self.tilings = 8
        self.alpha = 0.01/8
        self.iht = IHT(2048)
        self.shape = [8,8]
        self.delta=0.9

        self.last_state = None
        self.last_action = None
        self.last_tile =None
        self.weights =None
        self.values =None
        self.Z=None

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """

        self.weights = np.random.uniform(-0.001,0,2048)
        self.values = np.zeros([self.shape[0],self.shape[1],3])

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        tile = [self.shape[0]*(state[0]+1.2)/(1.2+0.5),self.shape[1]*(state[1]+0.07)/(0.07+0.07)]
        #self.action
        if np.random.uniform()<self.epsilon:
            self.action = np.random.randint(3)
        else:
            Q=[]
            #c = -float('inf')
            for self.action in range(3):
                tiles_action = tiles(self.iht,self.tilings,tile,[self.action])
                # ignore zero,sum w where not zero 
                Q.append(np.sum(self.weights[tiles_action]))
            Q = np.array(Q)
            self.action =  np.random.choice(np.where(Q == Q.max())[0])
        
        self.last_state = state
        self.last_action = self.action
        self.last_tile = tile      
        self.Z = np.zeros(len(self.weights))
        
        return self.action


    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        TD_error = reward
        for i in tiles(self.iht, self.tilings, self.last_tile, [self.last_action]):
            TD_error -=self.weights[i]
            self.Z[i]=1.0
            
        
        tile = [self.shape[0]*(state[0]+1.2)/(1.2+0.5),self.shape[1]*(state[1]+0.07)/(0.07+0.07)]
        
        if np.random.uniform()<self.epsilon:
            self.action = np.random.randint(3)
        else:
            Q=[]
            #c = -float('inf')
            for self.action in range(3):
                tiles_action = tiles(self.iht,self.tilings,tile,[self.action])
                # ignore zero,sum w where not zero 
                Q.append(np.sum(self.weights[tiles_action]))
            Q = np.array(Q)
            self.action =  np.random.choice(np.where(Q == Q.max())[0])        
        
        activated = np.zeros(len(self.weights))
        
        for i in tiles(self.iht, self.tilings, tile, [self.action]):
            activated[i]=1.0
            TD_error+=self.gamma*self.weights[i]
            
        
        self.weights += self.alpha*TD_error*self.Z
        self.Z = self.gamma * self.delta*self.Z
        self.last_state = state
        self.last_action = self.action
        self.last_tile = tile
        return self.action

        
        

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        TD_error =reward
        for i in tiles(self.iht, self.tilings, self.last_tile, [self.last_action]):
            TD_error -=self.weights[i]
            self.Z[i]=1.0
        
        self.weights+=self.alpha*TD_error*self.Z
        
        return
        
        
    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        pass