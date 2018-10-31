"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018,
  University of Alberta.
  Gambler's problem environment using RLGlue.
"""
from rl_glue import BaseEnvironment
import numpy as np


class WindygridEnvironment(BaseEnvironment):
    """
    Slightly modified Gambler environment -- Example 4.3 from
    RL book (2nd edition)

    Note: inherit from BaseEnvironment to be sure that your Agent class implements
    the entire BaseEnvironment interface
    """

    def __init__(self):
        """Declare environment variables."""
        self.width = 10
        self.height = 7
        self.wind_strength = [0,0,0,1,1,1,2,2,1,0]
        self.start = [0,3]
        self.goal = [7,3]

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """
        self.state = None

    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        Hint: Sample the starting state necessary for exploring starts and return.
        """
        self.state = np.array(self.start)
        return self.state

    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - terminal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """
        x = self.state[0] + action[0]
        y = self.state[1] + action[1]
        if x<0:
            x = 0
        if x>self.width-1:
            x = self.width-1
        if y<0:
            y=0
        if y>self.height-1:
            y= self.height -1
        
        y -= self.wind_strength[x]
        if y<0:
            y=0           
            
        self.state = np.array((x,y))
        reward = -1.0
        terminal = False
        if np.array_equal(self.state,self.goal):
            terminal = True
        return reward,self.state,terminal

    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
