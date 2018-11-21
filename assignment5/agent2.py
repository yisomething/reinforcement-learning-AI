"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
from tiles3 import*


class TileAgent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""

        self.alpha = 0.01/50
        self.tilings = 50
        self.iht = IHT(300)


    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """

        self.w = np.zeros(300)
        self.current_state = None
        self.last_state = None
        self.true_value = np.load("TrueValueFunction.npy")
        self.vector = {}

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.current_state = 500
        self.last_state = self.current_state
        self.action = np.random.randint(-100, 101)
        while self.action == 0:
            self.action = np.random.randint(-100, 101)
        return self.action


    def get_feature_vector(self,state):
        if state in self.vector:
            return self.vector[state]
        else:
            temp = np.zeros(300)
            mytiles = tiles(self.iht, self.tilings, [float(state)/200])
            for tile in mytiles:
                temp[tile] = 1
            self.vector[state] = temp
            return temp

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        self.current_state = state
        self.action = np.random.randint(-100, 101)
        while self.action == 0:
            self.action = np.random.randint(-100, 101)

        current_vector=self.get_feature_vector(self.current_state)
        last_vector=self.get_feature_vector(self.last_state)
        TD_error = self.alpha*(reward+np.dot(self.w,current_vector)-np.dot(self.w,last_vector))
        #TD_error  = self.alpha * (reward + self.w[self.current_state] - self.w[self.last_state])
        self.w += TD_error * last_vector

        self.last_state = self.current_state

        return self.action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        last_vector=self.get_feature_vector(self.last_state)
        TD_error = self.alpha * (reward - np.dot(self.w,last_vector))

        self.w += TD_error * last_vector

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if (in_message == 'RMSE'):
            estimate_value=[]
            for state in range(1,1001):
                vecotr=self.get_feature_vector(state)
                estimate_value.append(np.dot(self.w,vecotr))
            return np.sqrt(np.mean((self.true_value[1:] - estimate_value) ** 2))
        else:
            return "I dont know how to respond to this message!!"
