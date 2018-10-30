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
        row = self.state[0] + action[0]
        col = self.state[1] + action[1]
        if row<0:
            row = 0
        if row>self.width-1:
            row = self.width-1
        if col<0:
            col=0
        if col>self.height-1:
            col= self.height -1
        self.state[0] = row
        self.state[1] = col
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
