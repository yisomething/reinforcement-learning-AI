"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue
from env import RandomWalkEnvironment
from agent1 import TabularAgent
from agent2 import TileAgent
import numpy as np
from rndmwalk_policy_evaluation import *
import matplotlib.pyplot as plt
from tqdm import tqdm

def experiment(rlglue,num_episodes,num_runs):
    RMSE = np.zeros(num_episodes)
    for run in tqdm(range(num_runs)):

        np.random.seed(run)
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

    RMSE = [i / num_runs for i in RMSE]
    return RMSE


def main():
    num_episodes = 2000
    #max_steps = 10000
    num_runs = 2
    true_value = np.load("TrueValueFunction.npy")

    # Create and pass agent and environment objects to RLGlue
    environment = RandomWalkEnvironment()
    agent = TabularAgent()
    rlglue = RLGlue(environment, agent)
    result1 = experiment(rlglue,num_episodes,num_runs)
    del agent, environment  # don't use these anymore

    """
    environment = RandomWalkEnvironment()
    agent = TileAgent()
    rlglue = RLGlue(environment, agent)
    result2 = experiment(rlglue,num_episodes,num_runs)
    del agent, environment  # don't use these anymore

    """

    plt.plot(range(num_episodes),result1,label="TabularAgent")
    plt.xlabel("Episodes")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()



if __name__ == "__main__":
    main()
