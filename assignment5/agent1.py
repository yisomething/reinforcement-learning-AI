"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.

"""
from rl_glue import BaseAgent
import numpy as np

class TabularAgent(BaseAgent):
    """
    TabularAgent --

    Note: inherit from BaseAgent to be sure that your Agent class implements
    the entire BaseAgent interface
    """

    def __init__(self):
        """Declare agent variables."""
        self.alpha = 0.5
        self.action = None

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.w = np.zeros(1001) #weights
        self.current_state = None
        self.last_state = None
        self.true_value = np.load("TrueValueFunction.npy")
        self.vector={}


    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer

        """
        self.current_state = 500
        self.last_state = self.current_state
        self.action = np.random.randint(-100,101)
        while self.action == 0 :
            self.action = np.random.randint(-100,101)
        return self.action

    def get_feature_vector(self,state):
        if state in self.vector:
            return self.vector[state]
        else:
            features = np.zeros(1001)
            features[state] = 1
            self.vector[state]=features
            return features

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

        #in order to optimal running time, write self.w[state] as dot multiple result

        TD_error = self.alpha * (reward + self.w[self.current_state] - self.w[self.last_state])
        self.w += TD_error * self.get_feature_vector(self.last_state)
        self.last_state = self.current_state

        return self.action

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        TD_error = self.alpha * (reward - self.w[self.last_state])

        self.w += TD_error * self.get_feature_vector(self.last_state)


    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """


        if (in_message == 'RMSE'):

            return np.sqrt(np.mean((self.true_value - self.w) ** 2))
        else:
            return "I dont know how to respond to this message!!"
