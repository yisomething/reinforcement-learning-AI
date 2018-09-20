import numpy as np

from rl_glue import BaseEnvironment


class OneStateEnvironment(BaseEnvironment):
    """
    Example single-state environment with two actions
    """

    def __init__(self):
        """Declare environment variables."""
        super().__init__()

        self.reward_distribution = None

    def env_init(self):
        """
        Initialize environment variables.
        """
        # generate 10 random num with mean 0 and variance 1
        self.mean = 0
        self.var = 1
        self.num_actions = 10
        self.alist =[]
        for i in range(self.num_actions):
            self.alist.append(np.random.normal(self.mean,self.var))

        self.reward_distribution = self.alist.index(max(self.alist))


    def env_start(self):
        """
        The first method called when the experiment starts, called before the
        agent starts.

        Returns:
            The first state observation from the environment.
        """
        # only one state, which we will represent using 0
        return 0

    def env_step(self, action):
        """
        A step taken by the environment.

        Args:
            action: The action taken by the agent

        Returns:
            (float, state, Boolean): a tuple of the reward, state observation,
                and boolean indicating if it's terminal.
        """

        state = 0
        terminal = False
        reward = np.random.normal(self.alist[action],1)



        try:
            return reward, state, terminal
        except NameError:
            m = "Invalid action specified in One-State Environment's " \
                "env_step: {}"
            print(m.format(action))
            print("Please only return the integers 0 and 1 as actions.\n")
            exit(1)

    def env_message(self, message):
        pass
