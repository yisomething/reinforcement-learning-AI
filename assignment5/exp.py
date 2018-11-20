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
from tqdm import tqdm


if __name__ == "__main__":
    num_episodes = 2000
    #max_steps = 10000
    num_runs = 10

    # Create and pass agent and environment objects to RLGlue
    environment = RandomWalkEnvironment()
    agent = TabularAgent()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore

    true_value = np.load("TrueValueFunction.npy")

    for run in tqdm(range(num_runs)):


        # initialize RL-Glue
        rlglue.rl_init()

        RMSE = np.zeros(num_episodes)

        # loop over episodes
        for episode in tqdm(range(num_episodes)):

            # run episode with the allocated steps budget
            rlglue.rl_episode()
            #rlglue.rl(max_steps)

            # if episode is one of the key episodes, extract and save value
            # function
            #RMSE[episode] = rlglue.rl_agent_message("RMSE")



