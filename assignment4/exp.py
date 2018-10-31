"""
  Purpose: For use in the Reinforcement Learning course, Fall 2018, University of Alberta.
  Implementation of the interaction between the Gambler's problem environment
  and the Monte Carlo agent using RLGlue.
"""
from rl_glue import RLGlue
from env import WindygridEnvironment
from agent import SarsaAgent
import numpy as np

if __name__ == "__main__":
    num_episodes = 170
    max_steps = 10000
    num_runs = 10

    # Create and pass agent and environment objects to RLGlue
    environment = WindygridEnvironment()
    agent = SarsaAgent()
    rlglue = RLGlue(environment, agent)
    del agent, environment  # don't use these anymore
    for action in [8,9]:
        output = np.zeros((num_runs,num_episodes))

        rlglue.rl_agent_message("differ_action : {}".format(action))

        for run in range(num_runs):

            print("run number: {}\n".format(run))
            # set seed for reproducibility
            np.random.seed(run)

            # initialize RL-Glue
            rlglue.rl_init()



            # loop over episodes
            for episode in range(num_episodes):
                # run episode with the allocated steps budget
                rlglue.rl_episode(max_steps)

                output[run,episode] = rlglue.num_steps()

        average_over_runs = np.mean(output, axis=0)

        # save average value function numpy object, to be used by plotting script
        np.save("action_{}".format(action), average_over_runs)
