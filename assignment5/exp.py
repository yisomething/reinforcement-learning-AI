"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue
from env import RandomWalkEnvironment
from agent1 import TabularAgent
import numpy as np
import rndmwalk_policy_evaluation
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == "__main__":
    num_episodes = 2000
    #max_steps = 10000
    num_runs = 1

    # Create and pass agent and environment objects to RLGlue
    environment = RandomWalkEnvironment()
    agent = TabularAgent()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore

    true_value = np.load("TrueValueFunction.npy")
    RMSE = np.zeros(2000)
    data = np.zeros(2000)
    for run in tqdm(range(num_runs)):

        np.random.seed(num_runs)
        # initialize RL-Glue
        rlglue.rl_init()



        # loop over episodes
        for episode in tqdm(range(num_episodes)):

            # run episode with the allocated steps budget
            rlglue.rl_episode()
            #rlglue.rl(max_steps)

            # if episode is one of the key episodes, extract and save value
            # function
            RMSE[episode] += rlglue.rl_agent_message("RMSE")
            data[episode] = RMSE[episode]/2000


    plt.plot(range(2000),data)
    plt.show()




