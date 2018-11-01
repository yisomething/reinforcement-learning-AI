"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue
from env import WindygridEnvironment
from agent import SarsaAgent
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    max_steps = 8000
    num_runs = 1

    # Create and pass agent and environment objects to RLGlue
    environment = WindygridEnvironment()
    agent = SarsaAgent()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore
    for run in range(num_runs):
        episode=[]
        time_step=[]
        rlglue.rl_init()
        while True:
            rlglue.rl_episode()
            time_step.append(rlglue.num_steps())
            episode.append(rlglue.num_episodes())
            if rlglue.num_steps() > 8000:
                break

    plt.plot(time_step,episode,label="8 actions")
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
    plt.xlabel('Time steps')
    plt.ylabel('Episode', rotation=90)
    plt.legend(loc=2)
    plt.show()
        # save average value function numpy object, to be used by plotting script
