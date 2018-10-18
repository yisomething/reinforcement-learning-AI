"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np


class MonteCarloAgent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.Q = np.zeros((100, 100))
        self.pi = np.zeros(100)
        self.track = np.zeros((100, 100))
        self.total_values = np.zeros((100, 100))
        for s in range(100):
            self.pi[s] = min(s, 100 - s)

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.tmpstore = {}

        state = int(state[0])
        action = int(np.random.randint(1, min(state, 100 - state) + 1))
        self.tmpstore.update({(state, action): 0})
        return action

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        state = int(state[0])
        action = int(self.pi[state])
        self.tmpstore.update({(state, action):0})
        return action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        for key in self.tmpstore:
            self.track[key[0]][key[1]] += 1
            self.total_values[key[0]][key[1]] += reward
            self.Q[key[0]][key[1]] = self.total_values[key[0]][key[1]] / self.track[key[0]][key[1]]
            self.pi[key[0]] = np.argmax(self.Q[key[0]][1:]) + 1

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message == 'ValueFunction':
            return (np.max(self.Q, axis=1)).tostring()
        else:
            return "I dont know how to respond to this message!!"
