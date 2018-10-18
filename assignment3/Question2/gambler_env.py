"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018,
  University of Alberta.
  Gambler's problem environment using RLGlue.
"""
from rl_glue import BaseEnvironment
import numpy as np


class GamblerEnvironment(BaseEnvironment):
    """
    Slightly modified Gambler environment -- Example 4.3 from
    RL book (2nd edition)

    Note: inherit from BaseEnvironment to be sure that your Agent class implements
    the entire BaseEnvironment interface
    """

    def __init__(self):
        """Declare environment variables."""
        self.ph = 0.55
        self.num_states = 100
        self.randstate = None

    def env_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize environment variables necessary for run.
        """

    def env_start(self):
        """
        Arguments: Nothing
        Returns: state - numpy array
        Hint: Sample the starting state necessary for exploring starts and return.
        """
        self.randstate = np.random.randint(1,self.num_states)
        self.state = np.array([self.randstate])
        return self.state

    def env_step(self, action):
        """
        Arguments: action - integer
        Returns: reward - float, state - numpy array - reach_goal - boolean
        Hint: Take a step in the environment based on dynamics; also checking for action validity in
        state may help handle any rogue agents.
        """
        reach_goal = False
        reward = 0.0
        if np.random.uniform() < self.ph:
            self.state[0] = self.state[0] + action
        else:
            self.state[0] = self.state[0] - action

        if self.state[0] == self.num_states :
            reach_goal = True
            self.state = None
            reward = 1.0
        elif self.state[0] == 0:
            reach_goal = True
            self.state = None

        return reward, self.state, reach_goal
    def env_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: response based on in_message
        This function is complete. You do not need to add code here.
        """
        pass
