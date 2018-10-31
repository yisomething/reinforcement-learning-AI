"""
   Purpose: For use in the Reinforcement Learning course, Fall 2018,
   University of Alberta.
"""
from rl_glue import BaseAgent
import numpy as np


class SarsaAgent(BaseAgent):

    def __init__(self):
        """Declare agent variables."""
        self.width = 10
        self.height = 7
        self.alpha = 0.5
        self.epsilon = 0.1
        self.actions = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1),(0,0)]
        self.num_action = 8

    def agent_init(self):
        """
        Arguments: Nothing
        Returns: Nothing
        Hint: Initialize the variables that need to be reset before each run
        begins
        """
        self.Q = np.full((self.width,self.height,self.num_action),0.0)

    def agent_start(self, state):
        """
        Arguments: state - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        if np.random.random() > self.epsilon:
            self.current_action = np.argmax(self.Q[state[0],state[1]])
        else:
            self.current_action = np.random.randint(0,self.num_action)
        self.last_state = state
        self.last_action = self.current_action
        return np.array(self.actions[self.last_action])

    def agent_step(self, reward, state):
        """
        Arguments: reward - floting point, state - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        if np.random.random() > self.epsilon:
            self.current_action = np.argmax(self.Q[state[0],state[1]])
        else:
            self.current_action = np.random.randint(0,self.num_action)

        self.Q[self.last_state[0],self.last_state[1],self.last_action] += self.alpha*(reward + self.Q[state[0],state[1],self.current_action] -  self.Q[self.last_state[0],self.last_state[1],self.last_action])

        self.last_state = state
        self.last_action = self.current_action
        return np.array(self.actions[self.last_action])

    def agent_end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        self.Q[self.last_state[0],self.last_state[1],self.last_action] += self.alpha*(reward - self.Q[self.last_state[0],self.last_state[1],self.last_action])

    def agent_message(self, in_message):
        """
        Arguments: in_message - string
        Returns: The value function as a list.
        This function is complete. You do not need to add code here.
        """
        if in_message.startswith('differ_action'):
            self.num_action = int(in_message.split(":")[1])
        else:
            return "I dont know how to respond to this message!!"
