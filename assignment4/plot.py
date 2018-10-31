"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Plot script for Assignment 3 -- Gambler's problem with a Monte Carlo Exploring Starts agent.
"""

import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   for action in [8,9]:
      output = np.load('action_{}.npy'.format(action))
      print(output.shape)
      print(output)
      plt.plot(output,np.arange(170),label="{} actions".format(action))
   plt.yticks([0,50,100,150,170])
   plt.xticks([0,1000,2000,3000,4000,5000,6000,7000,8000])
   plt.xlabel('Time steps')
   plt.ylabel('Episode',rotation = 90)
   plt.legend(loc=2)
   plt.show()
