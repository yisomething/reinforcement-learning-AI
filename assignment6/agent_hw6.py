"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
from tile3 import*
import random

class Agent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)
    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.tilings = 8
        self.alpha = 0.1/8
        self.iht = IHT(2048)
        self.sizeTilings = [8,8]
        self.lmda=0.9

        self.last_state = None
        self.last_action = None
        self.last_tile =None
        self.weight =None
        self.Z=None

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """

        self.weight = np.random.uniform(-0.001,0,2048)

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.Z = np.zeros(2048)

        self.last_state = state
        self.last_action = self.action(state)

        return self.last_action


    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """

        action = self.action(state)
        delta = reward + self.q_value(state,action) - self.q_value(self.last_state,self.last_action)

        for index in self.get_index(self.last_state,self.last_action):
            delta = delta - self.weight[index]
            self.Z[index] = 1

        for index in self.get_index(state,action):
            delta = delta + self.weight[index]

        self.weight = self.weight + self.alpha*self.Z*delta
        self.Z = self.lmda * self.Z
        self.last_state = state
        self.last_action = action

        return action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """

        delta = reward-self.q_value(self.last_state, self.last_action)
        self.weight = self.weight + self.alpha * self.Z * delta

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        pass

    def get_index(self,state,action):

        tile = [self.sizeTilings[0]*(state[0])/(1.2+0.5),self.sizeTilings[1]*(state[1])/(0.07+0.07)]

        return tiles(self.iht, self.tilings,tile,[action])

    def q_value(self,state,action):


        tile = [self.sizeTilings[0]*(state[0])/(1.2+0.5),self.sizeTilings[1]*(state[1])/(0.07+0.07)]

        X = np.zeros(len(self.weight))

        for i in tiles(self.iht, self.tilings,tile,[action]):
            X[i] = 1.0

        return np.dot(self.weight,X)


    def action(self,state):

        tile = [self.sizeTilings[0]*(state[0])/(1.2+0.5),self.sizeTilings[1]*(state[1])/(0.07+0.07)]

        Q=np.zeros(3)
        for action in range(3):
            X = np.zeros(2048)
            for i in tiles(self.iht, self.tilings,tile,[action]):
                X[i] = 1.0
            Q[action] = np.dot(self.weight,X)
        
        return random.choice(random.choice(np.nonzero(Q == np.amax(Q))))

