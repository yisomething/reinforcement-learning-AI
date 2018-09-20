"""Example experiment for CMPUT 366 Fall 2019

This experiment uses the rl_step() function.

Runs a random agent in a 1D environment. Runs 10 (num_runs) iterations of
100 episodes, and reports the final average reward. Each episode is capped at
100 steps (max_steps).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from env import OneStateEnvironment
from agent1 import RandomAgent1
from agent2 import RandomAgent2
from rl_glue import RLGlue



def experiment2(rlg, num_runs, max_steps):

    rewards = np.zeros(max_steps)
    for run in range(num_runs):
        # set seed for reproducibility
        np.random.seed(run)

        # initialize RL-Glue
        rlg.rl_init()

        rlg.rl_start()

        for __ in range(max_steps):
            action= rlg.rl_step()[2]

            if action == rlg._environment.reward_distribution:
                rewards[__] +=1

    for i in range(max_steps):
        rewards[i] = rewards[i]/num_runs
    return rewards


def main():
    max_steps = 1000  # max number of steps in an episode
    num_runs = 2000  # number of repetitions of the experiment

    # Create and pass agent and environment objects to RLGlue
    agent = RandomAgent1()
    environment = OneStateEnvironment()
    rlglue = RLGlue(environment, agent)
    result1 = experiment2(rlglue,num_runs,max_steps)

    agent = RandomAgent2()
    environment = OneStateEnvironment()
    rlglue = RLGlue(environment, agent)
    result2 = experiment2(rlglue, num_runs, max_steps)


    axis = [i for i in range(max_steps)]

    plt.plot(axis, result1, color="grey")
    plt.plot(axis, result2, color="blue")

    plt.yticks([0,.2,.4,.6,.8,1],['0%','20%','40%','60%','80%','100%'])

    #formatter = FuncFormatter(str(100*y) + '%')
    #plt.gca().yaxis.set_major_formatter(formatter)
    plt.text(200,.85,r'Optimisic, greedy Q1=5, ε=0')
    plt.text(600,.55,r'Realistic,ε-greedy Q1=0, ε=0.1')
    plt.ylabel('%\nOptimal\naction',rotation = 0)
    plt.xlabel('Steps')
    plt.show()

if __name__ == '__main__':
    main()
