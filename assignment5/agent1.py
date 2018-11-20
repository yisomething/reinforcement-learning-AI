"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
   Monte Carlo agent using RLGlue - barebones.
"""
from rl_glue import BaseAgent
import numpy as np
import rndmwalk_policy_evaluation


class TabularAgent(BaseAgent):
    """
    Monte Carlo agent -- Section 5.3 from RL book (2nd edition)

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.alpha = 0.5
        self.w = None #weights
        self.current_state = None
        self.last_state = None
        self.action = None

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.w = np.zeros(1001)
        self.current_state = None
        self.last_state = None

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
        self.action = np.random.randint(-100,101)
        while self.action == 0 :
            self.action = np.random.randint(-100,101)
        return self.action

    def get_feature_vector(self,state):
        features = np.zeros(1001)
        features[state] = 1
        return features


    def value(self, state, weights):
        features = self.get_feature_vector(state)
        return np.dot(weights, features)

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        self.current_state = state
        self.action=np.random.randint(-100,101)
        while self.action == 0 :
            self.action = np.random.randint(-100,101)

        TD_error = self.alpha * (reward + self.value(self.current_state, self.w) - self.value(self.last_state, self.w))
        self.w = np.add(self.w, TD_error * self.get_feature_vector(self.last_state))
        self.last_state = self.current_state

        return self.action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        TD_error = self.alpha * (reward - self.value(self.last_state, self.w))

        self.w = np.add(self.w, TD_error * self.get_feature_vector(self.last_state))

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """

        if (in_message == 'RMSE'):
            estimated_values = np.zeros(1001)
            true_value = np.load("TrueValueFunction.npy")
            for state in range(1001):
                estimated_values[state] = self.value(state, self.w)

            return np.sqrt(np.mean((true_value[1:] - estimated_values[1:]) ** 2))
        else:
            return "I dont know how to respond to this message!!"
