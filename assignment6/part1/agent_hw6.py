"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html
"""
from rl_glue import BaseAgent
import numpy as np
from tile3 import*
import random

class Agent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL bstateook (2nd edition)
    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.epsilon =0
        self.gamma = 1.0
        self.tilings = 8
        self.alpha = 0.1/8
        self.iht = IHT(2048)
        self.sizeTilings = [8,8]
        self.lmda=0.9

        self.last_state = None
        self.last_action = None
        self.last_tile =None
        self.weights =None
        self.Z=None

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """

        self.weights = np.random.uniform(-0.001,0,2048)

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.Z = np.zeros(2048)
        tile = [self.sizeTilings[0]*(state[0])/(1.2+0.5),self.sizeTilings[1]*(state[1])/(0.07+0.07)]

        Q=np.zeros(3)
        for self.action in range(3):
            z_pointer = np.zeros(len(self.weights))
            for i in self.get_index(tile,self.action):
                z_pointer[i] = 1.0
            Q[self.action] = np.dot(self.weights,z_pointer)

        #choose action
        if np.random.uniform()<self.epsilon:
            self.action = np.random.randint(3)
        else:
            self.action=random.choice(random.choice(np.nonzero(Q == np.amax(Q))))


        self.last_state = state
        self.last_action = self.action
        self.last_tile = tile
        return self.action


    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        tile = [self.sizeTilings[0]*(state[0])/(1.2+0.5),self.sizeTilings[1]*(state[1])/(0.07+0.07)]
        for i in self.get_index(self.last_tile,self.last_action):
            reward -=self.weights[i]
            self.Z[i]=1.0

        Q=np.zeros(3)
        for self.action in range(3):
            z_pointer = np.zeros(len(self.weights))
            for i in tiles(self.iht, self.tilings,tile,[self.action]):
                z_pointer[i] = 1.0
                Q[self.action] = np.dot(self.weights,z_pointer)

        if np.random.uniform()<self.epsilon:
            self.action = np.random.randint(3)
        else:
            self.action =random.choice(random.choice(np.nonzero(Q == np.amax(Q))))


        for i in self.get_index(tile,self.last_action):
            reward +=self.gamma*self.weights[i]

        self.weights+=self.alpha*reward*self.Z
        self.Z = self.gamma*self.lmda*self.Z
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
        for i in self.get_index(self.last_tile,self.last_action):
            reward -=self.weights[i]
            self.Z[i]=1.0

        self.weights+=self.alpha*reward*self.Z

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if (in_message == '3D plot of the cast-to-go'):
            return self.weights

    def get_index(self,state, action):
        #using tiles to return corresponding index
        return tiles(self.iht, self.tilings, state, [action])
